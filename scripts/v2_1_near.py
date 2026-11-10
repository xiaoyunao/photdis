#!/usr/bin/env python3
"""Strict-PSF development near/far audit. Run only in an allocated Slurm job."""
import argparse
import os
import json
import numpy as np
from pathlib import Path
from v2_0_common import ROOT, load, subset, csv_rows, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.baseline_v0 import metrics
from photdis.probability_v2_1 import psf_admission, migration, interval_score

DEST = ROOT/'data/v2_1'
TABLES = DEST/'tables'
DISTANCES = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, np.inf)]
CLASSES = [(-1, 'unknown'), (0, 'dwarf'), (1, 'evolved'), (2, 'other')]


def strata(t, cross=True):
    n = len(t['identity'])
    out = {'all': np.ones(n, bool)}
    out.update({name: t['class_label'] == k for k, name in CLASSES})
    out.update(no_W1=~t['B'], W1_only=t['B'] & ~t['C'], W1W2=t['C'])
    mags = [(-np.inf, 18), (18, 20), (20, 22), (22, np.inf)]
    errors = [(0, .03), (.03, .1), (.1, np.inf)]
    for lo, hi in mags:
        out[f'r_{lo}_{hi}'] = (t['r'] >= lo) & (t['r'] < hi)
    for lo, hi in errors:
        out[f'error_{lo}_{hi}'] = (t['optical_error'] >= lo) & (t['optical_error'] < hi)
    for cell in np.unique(t['cell']):
        out[f'cell_{cell}'] = t['cell'] == cell
    for lo, hi in [(-np.inf, -1.5), (-1.5, -.5), (-.5, np.inf)]:
        out[f'FeH_{lo}_{hi}'] = (t['rvs_parameter_quality'] & np.isfinite(t['rvs_feh'])
                                & (t['rvs_feh'] >= lo) & (t['rvs_feh'] < hi))
        out[f'dwarf_FeH_{lo}_{hi}'] = out[f'FeH_{lo}_{hi}'] & (t['class_label'] == 0)
    if cross:
        for rlo, rhi in mags:
            for elo, ehi in errors:
                for wise in ('no_W1', 'W1_only', 'W1W2'):
                    name = f'r_{rlo}_{rhi}:error_{elo}_{ehi}:{wise}'
                    out[name] = out[f'r_{rlo}_{rhi}'] & out[f'error_{elo}_{ehi}'] & out[wise]
    return out


def reference(t, kind):
    mu = t['ref_'+kind].copy()
    sigma = t['sigma_mu'].copy()
    if kind in ('specdis', 'spdist'):
        mu[~t[kind+'_valid']] = np.nan
        sigma = t[kind+'_sigma_mu'].copy()
    if kind == 'Gaia':
        good = t['anchor_astrometry'] & ~t['nonstellar_risk'] & (t['p_corr'] > 0)
        mu[~good] = np.nan
        sigma = 5/np.log(10)*t['gaia_parallax_error']/np.maximum(t['p_corr'], 1e-20)
    return mu, sigma


def build():
    require_compute()
    TABLES.mkdir(parents=True, exist_ok=True)
    path = ROOT/'data/v2_0/chosen_expanded_A.npz'
    original = load(path)
    good, audit = psf_admission(original['TYPE'])
    assert not original['in_model_training'].any()
    t = subset(original, good)
    t['class_label'] = np.where(t['class_eligible'], t['class_label'], -1)
    assert (t['TYPE'] == 'PSF').all()
    atomic_npz(DEST/'development.npz', **t)
    counts, movements, boundary, gaia = [], [], [], []
    masks = strata(t)
    for split, role in [(1, 'cal'), (2, 'development_test')]:
        for name, group in masks.items():
            role_mask = (t['split'] == split) & group
            for kind in ('chosen', 'Gaia', 'specdis', 'spdist'):
                mu, sigma = reference(t, kind)
                valid = role_mask & np.isfinite(mu)
                distance = 10**((mu-10)/5)
                for lo, hi in DISTANCES:
                    use = valid & (distance >= lo) & (distance < hi)
                    row = dict(split=role, group=name, reference=kind, low_kpc=lo, high_kpc=hi,
                               **metrics(t['mu'][use], mu[use]))
                    row['interval_score68'] = float(np.mean(interval_score(t['mu'][use], mu[use]))) if use.any() else None
                    counts.append(row)
                selected = t['mu'][:, 2] < 15
                movements.append(dict(split=role, group=name, reference=kind, selection='median_lt10',
                                      **migration(mu[valid], selected[valid], sigma[valid])))
                known = np.isfinite(sigma) & (sigma > 0)
                crossing = known & (mu-sigma <= 15) & (mu+sigma >= 15)
                for label, cond in [('crossing_1sigma', crossing), ('certain_1sigma', known & ~crossing),
                                    ('unknown_error', ~known)]:
                    use = valid & cond
                    boundary.append(dict(split=role, group=name, reference=kind, boundary=label,
                                         **migration(mu[use], selected[use])))
            use = role_mask & t['parallax_eligible']
            pit = t['full_pit'][use]
            gaia.append(dict(split=role, group=name, n=int(use.sum()),
                             negative=int((t['p_corr'][use] < 0).sum()),
                             full_log_score=float(np.mean(t['full_log_density'][use])) if use.any() else None,
                             pit_tail=float(np.mean((pit < .05) | (pit > .95))) if use.any() else None))
    for name, rows in [('distance_class', counts), ('migration', movements),
                       ('boundary', boundary), ('gaia_observation', gaia)]:
        csv_rows(TABLES/(name+'.csv'), rows)
    # Reproduce historical claims before PSF admission; scientific rows above are new.
    use = (original['split'] == 2) & np.isfinite(original['ref_chosen'])
    hist = migration(original['ref_chosen'][use], original['mu'][use, 2] < 15)
    assert hist['near_selected']+hist['near_rejected'] == 43525
    assert hist['far_selected']+hist['far_rejected'] == 4231
    assert int((use & good).sum()) == 47740
    # The full reference inventory has already been inspected. Split3 is quarantine,
    # not unused final-validation sky. Do not fabricate an independent holdout.
    pop = load(ROOT/'data/v2_0/population.npz')
    pg, pa = psf_admission(pop['TYPE'])
    reserved = dict(status='acquisition_required', reserved_identities=0,
                    reason='Existing train/cal/test roles are used; split3 is legacy/identity/geometry quarantine.',
                    requirements=['New sky outside all historical train/cal/test and application-development footprints',
                                  'Exact Gaia/DESI/Legacy identity exclusion before scoring',
                                  'System membership and noise replicas kept in one role',
                                  'Verified DR11 PSF and independent label/observation provenance'],
                    existing_role_counts={str(k): int((pg & (pop['split'] == k)).sum()) for k in range(4)},
                    morphology=pa)
    save_json(DEST/'final_validation_reservation.json', reserved)
    app_path = ROOT/'data/v2_0/parent.npz'
    app = load(app_path)
    import fitsio
    app['mu'] = np.full((len(app['identity']), 5), np.nan)
    app['route'] = np.full(len(app['identity']), -1, dtype='i1')
    catalog_rows = []
    for field in np.unique(app['field']):
        use = app['field'] == field
        catalog = ROOT/'data/baseline_v1'/f'catalog_{field:02}.fits'
        with fitsio.FITS(str(catalog)) as f:
            a = f[1].read(rows=app['parent_row'][use], columns=['identity', 'ls_id', 'mu', 'route'])
        assert np.array_equal(a['identity'], app['identity'][use])
        assert np.array_equal(a['ls_id'], app['ls_id'][use])
        app['mu'][use], app['route'][use] = a['mu'], a['route']
        catalog_rows.append(dict(field=int(field), n=int(use.sum()), path=str(catalog.relative_to(ROOT))))
    ag, aa = psf_admission(app['TYPE'])
    predictable = ag & (app['route'] >= 0) & np.isfinite(app['mu'][:, 2])
    weight = app['sample_weight']
    near = predictable & (app['mu'][:, 2] < 15)
    application = dict(model='baseline_v1_recommended', morphology=aa,
                       sample_n=len(ag), psf_predictable_n=int(predictable.sum()),
                       near_sample_n=int(near.sum()),
                       weighted_psf_predictable=float(weight[predictable].sum()),
                       weighted_predicted_near=float(weight[near].sum()),
                       retention=float(weight[near].sum()/weight[predictable].sum()),
                       scope='Existing fixed26000-row stratified sample of13 DECam application fields; weighted parent8899334; northern no-model field excluded; not full DR11.',
                       catalogs=catalog_rows,
                       interpretation='Predicted count, not true near-star or class population count')
    atomic_npz(DEST/'application_psf.npz', **subset(app, ag))
    save_json(DEST/'application.json', application)
    save_json(DEST/'near_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'),
              input_sha256=sha(path), application_sha256=sha(app_path), morphology=audit,
              historical_reproduction=hist, display_clipping=False,
              exported_psf=bool((t['TYPE'] == 'PSF').all()), tables=len(counts)+len(movements)+len(boundary)+len(gaia)))
    print('NEAR_COMPLETE', json.dumps(dict(morphology=audit, historical=hist, application=application)), flush=True)


if __name__ == '__main__':
    build()

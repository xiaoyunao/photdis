#!/usr/bin/env python3
"""Cached scoring corrections and outcome-blind reference preparation."""
import json
import os
import numpy as np
from v2_1_near import ROOT, load, subset, csv_rows, save_json, sha, atomic_npz
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.probability_v2_1 import psf_admission
from baseline_v1_4_experiments import truth

DEST = ROOT/'data/v2_3'
TABLES = DEST/'tables'


def noise():
    old = ROOT/'data/v2_2'
    d = load(old/'noise_design.npz'); y = d['class_label'].astype(int)
    rows = []
    hashes = {str(old/'noise_design.npz'): sha(old/'noise_design.npz')}
    for replica in (-1, 0, 1, 2, 3):
        path = old/f'noise_chunks/replica_{replica}.npz'
        p = load(path); hashes[str(path)] = sha(path)
        assert np.array_equal(p['identity'], d['identity'])
        for lo in (18, 20, 22):
            use = (d['r_low'] == lo) & np.isin(y, [0, 1, 2])
            labels = y[use]; count = np.bincount(labels, minlength=3)
            # Design composition is known before noise/outcomes, not fitted to
            # predictions. Also give the strictly fixed uniform benchmark.
            design = count/count.sum()
            for method in ('prior_class', 'uniform', 'design_composition', 'point_class', 'averaged_class'):
                prob = np.tile(np.ones(3)/3 if method == 'uniform' else design, (len(labels), 1)) if method in ('uniform', 'design_composition') else p[method][use]
                score = np.log(np.maximum(prob[np.arange(len(labels)), labels], 1e-12))
                for c, group in [(None, 'all'), (0, 'dwarf'), (1, 'evolved'), (2, 'other')]:
                    take = np.ones(len(labels), bool) if c is None else labels == c
                    macro = np.mean([score[labels == k].mean() for k in range(3)])
                    rows.append(dict(replica=replica, r_low=lo, method=method, group=group, n=int(take.sum()),
                        log_score=float(score[take].mean()), macro_log_score=float(macro),
                        design_dwarf=float(design[0]), design_evolved=float(design[1]), design_other=float(design[2])))
    csv_rows(TABLES/'noise_baselines.csv', rows)
    save_json(DEST/'noise_correction.json', dict(inputs=hashes, simulation_rerun=False,
        paired_changes_preserved='manifests/v2_2/noise_cluster_bootstrap.csv',
        design_baseline='Known realized template class composition before scoring; descriptive design baseline, not estimated target-population prior',
        conclusion='Gain against a mismatched reference prior is not recovered faint color information.'))


def references(ids, paths):
    """Exact unique identity join; prefer new reference if present."""
    result = {}; seen = np.zeros(len(ids), bool)
    for path in paths:
        with np.load(path) as z:
            j = unique_join(ids, z['identity']); take = j >= 0
            for key in z.files:
                a = z[key]
                if a.ndim != 1 or len(a) != len(z['identity']):
                    continue
                if key not in result:
                    fill = np.nan if a.dtype.kind == 'f' else ('' if a.dtype.kind in 'SU' else 0)
                    result[key] = np.full(len(ids), fill, dtype=a.dtype)
                result[key][take] = a[j[take]]
            seen |= take
    assert seen.all() and np.array_equal(result['identity'], ids)
    return result


def qualify(t):
    t['class_label'] = np.where(t['class_eligible'], t['class_label'], -1)
    for kind in ('chosen', 'Gaia', 'specdis', 'spdist'):
        t['ref_'+kind] = truth(t, kind)
    t['weak_parallax_quality'] = t['parallax_eligible'] & (t['p_corr']/np.maximum(t['gaia_parallax_error'], 1e-30) < 5)
    t['distance_evaluation_eligible'] = (t['label_source'] > 0) & np.isfinite(t['ref_chosen']) & np.isfinite(t['sigma_mu']) & (t['sigma_mu'] > 0)
    return t


def main():
    require_compute(); TABLES.mkdir(parents=True, exist_ok=True)
    paths = [ROOT/'data/baseline_v1_3/references.npz', ROOT/'data/baseline_v1_4/references.npz']
    inputs = paths+[ROOT/'data/v2_1/frozen_training_audit.npz', ROOT/'data/v2_2/identity_unused_reservation.npz', ROOT/'docs/v2_3_protocol.md', ROOT/'scripts/v2_3_prepare.py']
    freeze = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    marker = DEST/'prepare_freeze.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == freeze
    else:
        save_json(marker, freeze)
    noise()
    reserve = load(ROOT/'data/v2_2/identity_unused_reservation.npz')
    t = qualify(references(reserve['identity'], [paths[1]]))
    t['TYPE'] = reserve['TYPE']; good, admission = psf_admission(t['TYPE']); assert good.all()
    assert not np.intersect1d(t['identity'], load(ROOT/'data/v2_1/development.npz')['identity']).size
    # Roles fixed before any predictions; every reserved ID remains evaluation.
    t['evaluation_role'] = np.full(len(good), 'same_sky_new_identity', dtype='U24')
    atomic_npz(DEST/'reserved_inputs.npz', **t)
    rows = []
    for lo, hi in [(-np.inf, 18), (18, 20), (20, 22), (22, 25), (25, np.inf)]:
        for wise in (False, True):
            u = (t['r'] >= lo) & (t['r'] < hi) & (t['B'] == wise)
            snr = t['p_corr']/np.maximum(t['gaia_parallax_error'], 1e-30)
            row = dict(r_low=lo, r_high=hi, W1=wise, n=int(u.sum()))
            for k in ('class_eligible','fit_eligible','distance_evaluation_eligible','specdis_valid','spdist_valid','anchor_astrometry','parallax_eligible','weak_parallax_quality'):
                row[k] = int((u & t[k]).sum())
            row.update(gaia_labels=int((u & (t['label_source'] == 1)).sum()), specdis_labels=int((u & (t['label_source'] == 2)).sum()),
                negative_parallax=int((u & t['parallax_eligible'] & (snr < 0)).sum()),
                parallax_snr_ge5=int((u & t['parallax_eligible'] & (snr >= 5)).sum()))
            for k in ('optical_error','sigma_mu','specdis_sigma_mu','spdist_sigma_mu'):
                valid = u & np.isfinite(t[k]) & (t[k] > 0)
                if k.startswith('specdis'): valid &= t['specdis_valid']
                if k.startswith('spdist'): valid &= t['spdist_valid']
                if k == 'sigma_mu': valid &= t['distance_evaluation_eligible']
                row[k+'_n'] = int(valid.sum())
                row[k+'_median'] = float(np.median(t[k][valid])) if valid.any() else None
            rows.append(row)
    csv_rows(TABLES/'reference_inventory.csv', rows)
    original = load(ROOT/'data/v2_1/frozen_training_audit.npz')
    psf, audit = psf_admission(original['TYPE'])
    train = qualify(references(original['identity'][psf], paths))
    train['TYPE'] = original['TYPE'][psf]
    # No reliable class requirement. Reject only unavailable measurement errors,
    # identically for control and candidate; retain every supported luminosity.
    valid = np.isfinite(train['Mg']) & np.isfinite(train['sigma_mu']) & (train['sigma_mu'] > 0)
    for k in ('sigma_g','sigma_r','sigma_z'):
        valid &= np.isfinite(train[k]) & (train[k] > 0)
    train = subset(train, valid)
    assert not np.intersect1d(train['identity'], t['identity']).size
    assert not np.intersect1d(train['identity'], load(ROOT/'data/v2_1/development.npz')['identity']).size
    atomic_npz(DEST/'training.npz', **train)
    save_json(DEST/'prepare_complete.json', dict(job=os.getenv('SLURM_JOB_ID'), frozen=freeze, reserved=len(t['identity']),
        training=len(train['identity']), training_morphology=audit, reserved_morphology=admission,
        unavailable_error_excluded=int((~valid).sum()), unlabeled_training=int((train['class_label'] < 0).sum()),
        luminosity_min=float(train['Mg'].min()), luminosity_max=float(train['Mg'].max()),
        roles='All reserved identities evaluation only, no calibration or fitting; no predictions read',
        outputs={name:sha(DEST/name) for name in ['training.npz','reserved_inputs.npz']}))
    print('PREPARE COMPLETE', len(train['identity']), len(t['identity']), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Frozen weak-flux PDF reuse and bounded photometric-systematic stress tests."""
import concurrent.futures
import multiprocessing as mp
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT, DEST, TABLES, DISTANCES, load, subset, csv_rows, save_json, sha, atomic_npz, reference
from v2_1_information import sample_indices
from baseline_v1_4_experiments import arrays
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.audit import rvs_metallicity
from photdis.baseline_v0 import metrics
from photdis.baseline_v1 import integrate
from photdis.baseline_v1_2 import full_parallax_log_density
from photdis.probability_v2_1 import near_mass, migration, interval_score

_MODEL = _T = None


def metallicity(t):
    corrected, domain = rvs_metallicity(t['rvs_feh'], t['rvs_teff'], t['rvs_logg'])
    quality = t['rvs_parameter_quality'] & np.isfinite(t['rvs_feh_err']) & (t['rvs_feh_err'] > 0)
    rows = []
    ref = t['ref_chosen']
    distance = 10**((ref-10)/5)
    for split, role in [(1, 'cal'), (2, 'development_test')]:
        for origin, feh, valid in [('raw_quality', t['rvs_feh'], quality),
                                   ('raw_correction_domain', t['rvs_feh'], quality & domain),
                                   ('corrected', corrected, quality & domain)]:
            for flo, fhi in [(-np.inf, -1.5), (-1.5, -.5), (-.5, np.inf)]:
                for lo, hi in DISTANCES:
                    use = ((t['split'] == split) & (t['class_label'] == 0) & valid & np.isfinite(feh)
                           & (feh >= flo) & (feh < fhi) & (distance >= lo) & (distance < hi))
                    rows.append(dict(split=role, abundance=origin, feh_low=flo, feh_high=fhi,
                                     low_kpc=lo, high_kpc=hi, **metrics(t['mu'][use], ref[use])))
    csv_rows(TABLES/'metallicity_quality.csv', rows)


def weak_flux(t):
    base = ROOT/'data/baseline_v1_6_wise'
    compact = load(base/'evaluation.npz')
    assert sha(base/'evaluation.npz') == json.loads((base/'result.json').read_text())['evaluation_sha256']
    edges = compact['edges']
    index = unique_join(t['identity'], compact['identity'])
    assert (index >= 0).all()
    qnear = np.full((len(t['identity']), 3), np.nan)
    zero = np.zeros((len(t['identity']), 3), bool)
    teacher = np.full_like(qnear, np.nan)
    for path in sorted((base/'chunks').glob('*.npz')):
        assert sha(path) == json.loads(path.with_suffix('.json').read_text())['sha256']
        p = load(path)
        j = unique_join(p['identity'], t['identity'])
        good = j >= 0
        rows = j[good]
        pdf = p['pdf'][good]
        target = t['g0'][rows]-t['ref_chosen'][rows]
        bins = np.searchsorted(edges, np.nan_to_num(target, nan=-1e9), side='right')-1
        inside = np.isfinite(target) & (bins >= 0) & (bins < len(edges)-1)
        for arm in range(3):
            qnear[rows, arm] = near_mass(pdf[:, arm], edges, t['g0'][rows])
            density = np.zeros(len(rows))
            density[inside] = pdf[np.flatnonzero(inside), arm, bins[inside]]/np.diff(edges)[bins[inside]]
            with np.errstate(divide='ignore'):
                teacher[rows, arm] = np.log(density)
            teacher[rows[~np.isfinite(target)], arm] = np.nan
            zero[rows, arm] = np.isfinite(target) & (density == 0)
    assert np.isfinite(qnear).all()
    state, flux = compact['state'][index], compact['flux'][index]
    masks = dict(all=np.ones(len(index), bool), W1_weak=state[:, 0] == 3,
                 W2_weak_given_W1=(state[:, 1] == 3) & (state[:, 0] >= 3),
                 W2_weak_no_W1=(state[:, 1] == 3) & (state[:, 0] < 3),
                 signed_nonpositive=((state >= 3) & (flux <= 0)).any(axis=1))
    masks.update({name+':r18_20': mask & (t['r'] >= 18) & (t['r'] < 20) for name, mask in list(masks.items())})
    rows = []
    for role, split in [('cal', 1), ('development_test', 2)]:
        for group, mask in masks.items():
            for c, label in [(None, 'all'), (0, 'dwarf'), (1, 'evolved'), (2, 'other'), (-1, 'unknown')]:
                use = mask & (t['split'] == split) & np.isfinite(t['ref_chosen'])
                if c is not None:
                    use &= t['class_label'] == c
                near = t['ref_chosen'][use] < 15
                for arm, name in enumerate(['optical', 'signed_W1', 'signed_W1W2']):
                    score = teacher[use, arm]
                    for threshold in [.5, .8, .95]:
                        rows.append(dict(split=role, group=group, population=label, model=name, threshold=threshold,
                            brier=float(np.mean((qnear[use, arm]-near)**2)) if use.any() else None,
                            log_teacher=float(score.mean()) if len(score) and np.isfinite(score).all() else None,
                            zero_teacher_density=int(zero[use, arm].sum()),
                            **migration(t['ref_chosen'][use], qnear[use, arm] > threshold, t['sigma_mu'][use])))
    csv_rows(TABLES/'weak_flux_probability.csv', rows)
    atomic_npz(DEST/'weak_flux_probability.npz', identity=t['identity'], TYPE=t['TYPE'], qnear=qnear,
               log_teacher=teacher, state=state, signed_flux=flux)


def systematic(task):
    scenario, sign = task
    m, e, ids = arrays(_T, np.arange(len(_T['identity'])))
    if scenario in ('g_zero_point', 'r_zero_point', 'z_zero_point'):
        m[:, {'g_zero_point': 0, 'r_zero_point': 1, 'z_zero_point': 2}[scenario]] += sign*.01
    elif scenario == 'dust_scale':
        for b, band in enumerate(['G', 'R', 'Z']):
            m[:, b] += sign*.1*2.5*np.log10(_T['MW_TRANSMISSION_'+band])
    elif scenario == 'error_scale':
        e[:, :3] *= 1+sign*.2
    elif scenario == 'sky_color_offset':
        # Deliberate coherent cell perturbation, not estimated instrumental error.
        offset = np.where(_T['cell'] % 2 == 0, 1., -1.)*.01
        m[:, 0] += offset
        m[:, 1] -= offset
    p = integrate(_MODEL, m, e, ids, draws=128)['wide']
    qnear = near_mass(p['pdf'], _MODEL.edges, m[:, 0])
    ref = _T['ref_chosen']
    use = np.isfinite(ref)
    ga = _T['parallax_eligible']
    log = full_parallax_log_density(p['pdf'][ga], _MODEL.edges, m[ga, 0],
                                     _T['p_corr'][ga], _T['gaia_parallax_error'][ga], order=16)
    out = dict(scenario=scenario, sign=sign, n_sample=len(ids),
               **metrics(p['mu'][use], ref[use]),
               qnear_mean=float(qnear.mean()),
               brier=float(np.mean((qnear[use]-(ref[use] < 15))**2)),
               interval_score68=float(np.mean(interval_score(p['mu'][use], ref[use]))),
               gaia_n=int(ga.sum()), gaia_negative=int((ga & (_T['p_corr'] < 0)).sum()),
               gaia_observation_log_score=float(log.mean()))
    atomic_npz(DEST/f'systematic_{scenario}_{sign}.npz', identity=ids, TYPE=_T['TYPE'], mu=p['mu'], qnear=qnear)
    return out


def main():
    global _MODEL, _T
    require_compute()
    t = load(DEST/'development.npz')
    assert (t['TYPE'] == 'PSF').all()
    metallicity(t)
    weak_flux(t)
    print('QUALITY cached weak PDFs and metallicity complete', flush=True)
    indices = np.flatnonzero(t['split'] == 2)
    indices = indices[sample_indices(t['identity'][indices], 512)]
    _T = subset(t, indices)
    _MODEL = joblib.load(ROOT/'data/baseline_v1_4/experiments/expanded_A_distance.joblib')
    tasks = [('nominal', 0), ('sky_color_offset', 1)]
    tasks += [(name, sign) for name in ('g_zero_point', 'r_zero_point', 'z_zero_point', 'dust_scale', 'error_scale') for sign in (-1, 1)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('fork')) as pool:
        scores = list(pool.map(systematic, tasks))
    csv_rows(TABLES/'systematic_sensitivity.csv', scores)
    save_json(DEST/'quality_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'), n_systematics=len(indices),
        selection='Identity hash, development test only; no residual-based selection',
        scopes=dict(metallicity='Raw, matched domain, author-corrected FeH; finite positive formal error',
                    weak_flux='Frozen legacy template audit on strict PSF evaluation, raw signed flux retained',
                    systematic='Prespecified stress amplitudes, not measured sky-error distribution or calibration'),
        no_new_model=True, no_final_validation=True))


if __name__ == '__main__':
    main()

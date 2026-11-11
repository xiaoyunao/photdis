#!/usr/bin/env python3
"""Actual same-identity source-separated scores for v3 and its QRF control."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import numpy as np
from v2_1_near import ROOT, load, sha, atomic_npz, save_json, csv_rows
from photdis.stage1 import require_compute
from photdis.audit import rvs_metallicity
from photdis.v3.controls import optical_magnitudes, probability_metrics, reliability_rows
from photdis.v3.metrics import (observed_parallax, normal_reference_score, distance_summary,
                              histogram_parallax, histogram_reference_score)
D = ROOT/'data/v3_0'
_EVIDENCE = _CAL = _SOURCE = _ARGS = _DEST = _FROZEN = None


def evidence_views(t):
    views = {}
    for c, name in enumerate(('specdis', 'spdist')):
        distance = t[name+'_distance']; sigma = t[name+'_sigma_mu']
        valid = t[name+'_valid'] & np.isfinite(distance) & (distance > 0) & np.isfinite(sigma) & (sigma > 0)
        mu = np.full(len(distance), np.nan)
        mu[valid] = 10+5*np.log10(distance[valid])+_CAL['map_parameters'][c, 0]
        error = np.hypot(sigma, _CAL['map_parameters'][c, 1])
        views[name] = (mu, error, valid, np.full(len(mu), _CAL['map_parameters'][c, 2]))
    p = t['p_corr']; pe = t['gaia_parallax_error']
    valid = t['parallax_eligible'] & (p > 10*pe)
    mu = np.full(len(p), np.nan); mu[valid] = 10-5*np.log10(p[valid])
    error = np.full(len(p), np.nan); error[valid] = 5/np.log(10)*pe[valid]/p[valid]
    views['gaia_precise'] = (mu, error, valid, np.zeros(len(p)))
    chosen_mu = np.full(len(p), np.nan); chosen_error = chosen_mu.copy(); chosen_fraction = np.zeros(len(p))
    for c, name in enumerate(('specdis', 'spdist', 'gaia_precise'), 1):
        mu, error, valid, fraction = views[name]
        use = (t['primary_channel'] == c) & valid
        chosen_mu[use] = mu[use]; chosen_error[use] = error[use]; chosen_fraction[use] = fraction[use]
    views['primary'] = (chosen_mu, chosen_error, np.isfinite(chosen_mu), chosen_fraction)
    return views


def evaluate_chunk(path):
    start = time.monotonic(); summary = load(path)
    marker = _DEST/path.with_suffix('.json').name; output = marker.with_suffix('.npz')
    if marker.exists():
        old = json.loads(marker.read_text()); assert old['inputs'] == _FROZEN
        assert sha(output) == old['sha256']; return str(output)
    order = np.argsort(_EVIDENCE['identity'])
    selected = order[np.searchsorted(_EVIDENCE['identity'][order], summary['identity'])]
    np.testing.assert_array_equal(_EVIDENCE['identity'][selected], summary['identity'])
    t = {k: v[selected] for k, v in _EVIDENCE.items()}; views = evidence_views(t)
    assert (summary['TYPE'] == 'PSF').all() and (t['TYPE'] == 'PSF').all()
    n = len(selected); gp = np.full(n, np.nan); gs = gp.copy()
    scores = {name: gp.copy() for name in views}
    if _ARGS.view == 'qrf':
        density = None
        edges = summary['g0'][:, None]-summary['Mg_edges'][None, ::-1]
        pdf = summary['pdf'][:, ::-1]
        qnear = np.sum(pdf*np.clip((15-edges[:, :-1])/np.diff(edges), 0, 1), axis=1)
    else:
        density = load(path.with_name(path.stem+'_joint.npz')); qnear = summary['qnear']
    for i in range(n):
        if density is not None:
            lo, hi = density['full_grid_offset'][i:i+2]
            grid = density['full_mu'][lo:hi]; logpdf = density['full_logpdf'][lo:hi]
        if t['parallax_eligible'][i]:
            obs = (observed_parallax(grid, logpdf, t['p_corr'][i], t['gaia_parallax_error'][i])
                   if density is not None else histogram_parallax(edges[i], pdf[i], t['p_corr'][i], t['gaia_parallax_error'][i], order=32))
            gp[i], gs[i] = obs['pit'], obs['log_score']
        for name, (mu, error, valid, fraction) in views.items():
            if not valid[i]: continue
            primary_source = {1: 'specdis', 2: 'spdist'}.get(int(t['primary_channel'][i])) if name == 'primary' else name
            broad_sigma = np.hypot(t[primary_source+'_sigma_mu'][i], 2.) if primary_source in ('specdis', 'spdist') else 2.
            if density is not None:
                scores[name][i] = normal_reference_score(grid, logpdf, mu[i], error[i],
                    broad_fraction=fraction[i], broad_error=broad_sigma)
            else:
                scores[name][i] = histogram_reference_score(edges[i], pdf[i], mu[i], error[i],
                    broad_fraction=fraction[i], broad_error=broad_sigma)
    out = dict(identity=summary['identity'], TYPE=summary['TYPE'], rows=selected,
        mu=summary['mu'], qnear=qnear, supported=summary['supported'], gaia_pit=gp, gaia_logscore=gs)
    if density is not None:
        out.update(class_probability=summary['class_distance'].sum(axis=2),
            class_distance=summary['class_distance'], negative_intrinsic_support=summary['negative_intrinsic_support'],
            discrepancy_probability=summary['discrepancy_probability'], converged=summary['converged'])
    for name, (mu, error, valid, _) in views.items():
        out[name+'_mu'] = mu; out[name+'_error'] = error; out[name+'_valid'] = valid; out[name+'_logscore'] = scores[name]
    atomic_npz(output, **out)
    save_json(marker, dict(inputs=_FROZEN, sha256=sha(output), n=n, elapsed_seconds=time.monotonic()-start))
    return str(output)


def main():
    global _EVIDENCE, _CAL, _SOURCE, _ARGS, _DEST, _FROZEN
    require_compute(); parser = argparse.ArgumentParser()
    parser.add_argument('cohort', choices=['hyper', 'development', 'application', 'reservation', 'new_sky', 'independent_scale'])
    parser.add_argument('--view', choices=['qrf', 'grz', 'griz', 'multiband'], default='multiband')
    parser.add_argument('--limit', type=int, default=0); parser.add_argument('--workers', type=int, default=32)
    args = _ARGS = parser.parse_args()
    folder = f'{args.cohort}_qrf_{args.limit}' if args.view == 'qrf' else f'{args.cohort}_{args.view}_fit_{args.limit}'
    final = args.cohort in ('reservation', 'new_sky', 'independent_scale')
    if final:
        from v3_0_validation_inputs import check_freeze
        method = check_freeze(); assert args.view == 'multiband' and args.limit == 0
        validation = D/'validation'/args.cohort
        evidence_marker = json.loads((validation/'evidence.json').read_text())
        assert evidence_marker['method_freeze'] == method
        prediction = validation/'predictions'; evidence_path = validation/'evidence.npz'; source_path = validation/'photometry.npz'
        assert sha(evidence_path) == evidence_marker['sha256']
        folder = 'final_'+folder
    else:
        prediction = D/'predictions'/folder; evidence_path = D/'evaluation'/(args.cohort+'.npz'); source_path = D/'cohorts'/(args.cohort+'.npz')
    assert (prediction/'complete.json').exists(), 'All prediction chunks required'
    paths = [prediction/'complete.json', evidence_path,
        source_path, D/'calibration.npz', ROOT/'scripts/v3_0_evaluate.py',
        ROOT/'src/photdis/v3/metrics.py', ROOT/'src/photdis/v3/controls.py']
    _FROZEN = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    _DEST = D/'evaluation_results'/folder; _DEST.mkdir(parents=True, exist_ok=True)
    freeze = _DEST/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == _FROZEN
    else: save_json(freeze, _FROZEN)
    _EVIDENCE = load(paths[1]); _SOURCE = load(paths[2]); _CAL = load(paths[3])
    np.testing.assert_array_equal(_EVIDENCE['identity'], _SOURCE['identity'])
    chunks = [p for p in sorted(prediction.glob('chunk_*.npz')) if not p.name.endswith('_joint.npz')]
    for path in chunks:
        meta = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == meta['sha256' if args.view == 'qrf' else 'summary_sha256']
        if args.view != 'qrf': assert sha(path.with_name(path.stem+'_joint.npz')) == meta['joint_sha256']
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        outputs = list(pool.map(evaluate_chunk, chunks))
    records = [load(p) for p in outputs]
    result = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
    rows = result['rows']; t = {k: v[rows] for k, v in _EVIDENCE.items()}
    phot = {k: v[rows] for k, v in _SOURCE.items()}; mag, error = optical_magnitudes(phot)
    feh, valid_feh = rvs_metallicity(t['rvs_feh'], t['rvs_teff'], t['rvs_logg'])
    groups = dict(all=np.ones(len(rows), bool), supported=result['supported'], qnear95=result['qnear'] > .95,
        red=(mag[:, 0]-mag[:, 1]) > 1.2, poor_dwarf=valid_feh & t['rvs_parameter_quality'] & (feh < -1.5) & (t['class_label'] == 0),
        intermediate_metallicity=valid_feh & (feh >= -1.5) & (feh < -.5),
        no_W1=~phot['band_mask'][:, 4], weak_W1=phot['band_state'][:, 4] == 3,
        high_error=error.max(axis=1) >= .1)
    groups['historical_fit'] = t['historical_fit']
    groups['not_historical_fit'] = ~t['historical_fit']
    if 'current_SED_fit' in t:
        groups['current_SED_fit'] = t['current_SED_fit']
        groups['not_current_SED_fit'] = ~t['current_SED_fit']
    groups.update({name: t['class_label'] == c for c, name in [(-1, 'unknown'), (0, 'dwarf'), (1, 'evolved'), (2, 'other')]})
    r = mag[:, 1]-2.5*np.log10(phot['transmission'][:, 1])
    groups['observable'] = (r < 20) & (error.max(axis=1) < .1) & result['supported']
    groups['near_observable'] = groups['observable'] & (result['qnear'] > .95)
    if 'current_SED_fit' in t:
        groups['near_observable_not_fit'] = groups['near_observable'] & ~t['current_SED_fit']
    for lo, hi in [(0, 18), (18, 20), (20, 22), (22, 99)]: groups[f'r_{lo}_{hi}'] = (r >= lo) & (r < hi)
    for cell in np.unique(phot['cell']): groups[f'sky_{cell}'] = phot['cell'] == cell
    distance_rows = []; gaia_rows = []; class_rows = []
    for group, use in groups.items():
        gaia = use & np.isfinite(result['gaia_logscore'])
        weight = phot['design_weight'][gaia]
        gaia_rows.append(dict(weight_sum=float(weight.sum()),
            effective_n=float(weight.sum()**2/(weight@weight)) if gaia.any() else None,
            weighted_log_score=float(np.average(result['gaia_logscore'][gaia], weights=weight)) if gaia.any() else None,
            weighted_pit_low=float(np.average(result['gaia_pit'][gaia] < .05, weights=weight)) if gaia.any() else None,
            weighted_pit_high=float(np.average(result['gaia_pit'][gaia] > .95, weights=weight)) if gaia.any() else None,
            group=group, n=int(gaia.sum()), negative=int((gaia & (t['p_corr'] < 0)).sum()),
            log_score=float(result['gaia_logscore'][gaia].mean()) if gaia.any() else None,
            pit_low=float((result['gaia_pit'][gaia] < .05).mean()) if gaia.any() else None,
            pit_high=float((result['gaia_pit'][gaia] > .95).mean()) if gaia.any() else None))
        for source in ('primary', 'specdis', 'spdist', 'gaia_precise'):
            valid = use & result[source+'_valid']
            for low, high in [(0, np.inf), (0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, np.inf)]:
                dist = 10**((result[source+'_mu']-10)/5)
                selected = valid & (dist >= low) & (dist < high)
                distance_rows.append(dict(group=group, reference=source, low_kpc=low, high_kpc=high,
                    **distance_summary(result['mu'][selected], result[source+'_mu'][selected]),
                    log_score=float(result[source+'_logscore'][selected].mean()) if selected.any() else None))
        if 'class_probability' in result:
            known = use & (t['class_label'] >= 0)
            class_rows.append(dict(group=group, **probability_metrics(result['class_probability'][known], t['class_label'][known])))
    atomic_npz(_DEST/'per_star.npz', **result, class_label=t['class_label'], corrected_feh=feh,
        corrected_feh_valid=valid_feh & t['rvs_parameter_quality'], r=r, gr=mag[:, 0]-mag[:, 1], rz=mag[:, 1]-mag[:, 2],
        cell=phot['cell'], RA=phot['RA'], DEC=phot['DEC'])
    csv_rows(_DEST/'distance.csv', distance_rows); csv_rows(_DEST/'gaia.csv', gaia_rows)
    if class_rows:
        csv_rows(_DEST/'class.csv', class_rows)
        known = t['class_label'] >= 0
        csv_rows(_DEST/'class_reliability.csv', reliability_rows(result['class_probability'][known], t['class_label'][known]))
    save_json(_DEST/'complete.json', dict(inputs=_FROZEN, rows=len(rows),
        group_count=len(groups), chunks=len(chunks), job=os.getenv('SLURM_JOB_ID'),
        outputs={p.name: sha(p) for p in _DEST.iterdir() if p.suffix in ('.npz', '.csv')},
        scope='Seen development/application, design-weighted Gaia columns describe available evidence only; un-clipped metrics; reference source scales not independent truths; Gaia association includes signed observations.'))
    print('EVALUATION COMPLETE', folder, len(rows), flush=True)


if __name__ == '__main__': main()

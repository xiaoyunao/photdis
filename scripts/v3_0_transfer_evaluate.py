#!/usr/bin/env python3
"""Paired prospective transfer scores with explicit omitted-response bounds.

The physical likelihood/metrics are reused from the frozen evaluator. This driver
changes only product routing and adds rigorous bounds for global response pruning.
"""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
from types import SimpleNamespace
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3.metrics import distance_summary
from v3_0_validation_inputs import check_freeze
import v3_0_evaluate as evaluate
D = ROOT/'data/v3_0'


def gaussian_score_bounds(retained_logscore, sigma, log_discarded_mass):
    """Bound a mixture's Gaussian-observation density, including tiny weights."""
    score, sigma = np.broadcast_arrays(np.asarray(retained_logscore), np.asarray(sigma))
    if np.any(sigma <= 0):
        raise ValueError('Positive observation errors required')
    if log_discarded_mass is None:
        return score.copy(), score.copy()
    if not log_discarded_mass < 0:
        raise ValueError('Omitted mass must be below one')
    lower = score + np.log1p(-np.exp(log_discarded_mass))
    upper = np.logaddexp(lower, log_discarded_mass - np.log(np.sqrt(2*np.pi)*sigma))
    return lower, upper


def main():
    require_compute()
    p = argparse.ArgumentParser(); p.add_argument('cohort', choices=['reservation','new_sky','independent_scale'])
    p.add_argument('--workers', type=int, default=32); a = p.parse_args()
    method = check_freeze(); plan_path = D/'catalog/transfer_plan.json'
    plan = json.loads(plan_path.read_text()); assert plan['method_freeze'] == method
    # Current concentrated finite response has one retained model. Multiple
    # retained responses require an explicit mixture product before this driver.
    assert len(plan['responses']) == 1, 'Combine retained response products first'
    response = plan['responses'][0]
    base = D/'validation'/a.cohort
    target = base/'population_transfer'/f'refit_{response["draw"]:02}_s{response["draws"]}'
    target_record = json.loads((target/'complete.json').read_text())
    assert target_record['integration_passed'] and target_record['inputs']['method_freeze'] == method
    em = json.loads((base/'evidence.json').read_text()); assert em['method_freeze'] == method
    assert sha(base/'evidence.npz') == em['sha256']
    evidence, source = load(base/'evidence.npz'), load(base/'photometry.npz')
    np.testing.assert_array_equal(evidence['identity'], source['identity'])
    assert (source['TYPE'] == 'PSF').all()
    out = D/'transfer_evaluation'/a.cohort; out.mkdir(parents=True, exist_ok=True)
    inputs = {str(x.relative_to(ROOT)): sha(x) for x in [plan_path, target/'complete.json',
        base/'evidence.json', base/'photometry.json', D/'calibration.npz', ROOT/'scripts/v3_0_evaluate.py',
        ROOT/'scripts/v3_0_transfer_evaluate.py', ROOT/'src/photdis/v3/metrics.py']}
    results = {}
    for view, folder in [('matched_reference', target/'reference_prediction'), ('target', target)]:
        destination = out/view; destination.mkdir(exist_ok=True)
        evaluate._EVIDENCE, evaluate._SOURCE, evaluate._CAL = evidence, source, load(D/'calibration.npz')
        evaluate._ARGS = SimpleNamespace(view='multiband', target_product=view == 'target')
        evaluate._DEST, evaluate._FROZEN = destination, dict(inputs, view=view)
        chunkdir = folder/'chunks' if view == 'target' else folder
        paths = sorted(x for x in chunkdir.glob('chunk_*.npz') if not x.stem.endswith(('_joint','_counts')))
        for path in paths:
            meta = json.loads(path.with_suffix('.json').read_text())
            assert sha(path) == (meta['outputs'][path.name] if view == 'target' else meta['summary_sha256'])
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('fork')) as pool:
            records = [load(x) for x in pool.map(evaluate.evaluate_chunk, paths)]
        results[view] = {k:np.concatenate([r[k] for r in records]) for k in records[0]}
    target_values = results['target']; row = target_values['rows']; n = len(row)
    assert len(np.unique(row)) == len(source['identity']) == n
    for result in results.values(): np.testing.assert_array_equal(result['identity'], target_values['identity'])
    phot = {k:v[row] for k,v in source.items()}; t = {k:v[row] for k,v in evidence.items()}
    mag, err = optical_magnitudes(phot); r = mag[:,1]-2.5*np.log10(phot['transmission'][:,1])
    observable = (r < 20) & (err.max(axis=1) < .1) & target_values['supported']
    groups = dict(all=np.ones(n,bool), observable=observable, target_near95=target_values['qnear']>.95,
        target_near_observable=observable & (target_values['qnear']>.95),
        reference_near_observable=observable & (results['matched_reference']['qnear']>.95),
        red=mag[:,0]-mag[:,1]>1.2)
    for lo,hi in [(0,18),(18,20),(20,22),(22,99)]: groups[f'r_{lo}_{hi}'] = (r>=lo)&(r<hi)
    gaia_rows, distance_rows, paired = [], [], []
    for view, result in results.items():
        finite = np.isfinite(result['gaia_logscore'])
        lower, upper = np.full(n,np.nan), np.full(n,np.nan)
        lower[finite], upper[finite] = gaussian_score_bounds(result['gaia_logscore'][finite],
            t['gaia_parallax_error'][finite], plan['discarded_response_log_mass'] if view=='target' else None)
        for group, mask in groups.items():
            use = mask & finite; w = phot['design_weight'][use]
            def avg(x): return float(np.average(x[use],weights=w)) if use.any() else None
            delta = plan['discarded_response_mass'] if view=='target' else 0.
            gaia_rows.append(dict(view=view,group=group,n=int(use.sum()),negative=int((use&(t['p_corr']<0)).sum()),
                weighted_logscore_retained=avg(result['gaia_logscore']),weighted_logscore_lower=avg(lower),weighted_logscore_upper=avg(upper),
                pit_low=avg(result['gaia_pit']<.05),pit_high=avg(result['gaia_pit']>.95),
                pit_low_lower=avg(result['gaia_pit']+delta<.05),pit_low_upper=avg(result['gaia_pit']-delta<.05),
                pit_high_lower=avg(result['gaia_pit']-delta>.95),pit_high_upper=avg(result['gaia_pit']+delta>.95),
                weight_sum=float(w.sum()),effective_n=float(w.sum()**2/(w@w)) if len(w) else None))
            for reference in ['primary','specdis','spdist','gaia_precise']:
                distance = 10**((result[reference+'_mu']-10)/5)
                for lo,hi in [(0,np.inf),(0,2),(2,5),(5,10),(10,20),(20,50),(50,np.inf)]:
                    selected=mask&result[reference+'_valid']&(distance>=lo)&(distance<hi)
                    distance_rows.append(dict(view=view,group=group,reference=reference,low_kpc=lo,high_kpc=hi,
                        **distance_summary(result['mu'][selected],result[reference+'_mu'][selected])))
        atomic_npz(out/(view+'.npz'),**result,gaia_logscore_lower=lower,gaia_logscore_upper=upper)
    for group, mask in groups.items():
        use = mask & np.isfinite(target_values['gaia_logscore'])
        gain = target_values['gaia_logscore']-results['matched_reference']['gaia_logscore']
        paired.append(dict(group=group,n=int(use.sum()),gaia_mean_logscore_gain=float(gain[use].mean()) if use.any() else None,
            median_abs_mu50_shift=float(np.median(abs(target_values['mu'][mask,2]-results['matched_reference']['mu'][mask,2]))) if mask.any() else None,
            selection='Same identities selected by named fixed rule; source response/calibration identical'))
    csv_rows(out/'gaia.csv',gaia_rows);csv_rows(out/'distance.csv',distance_rows);csv_rows(out/'paired.csv',paired)
    save_json(out/'complete.json',dict(inputs=inputs,method_freeze=method,n=n,job=os.getenv('SLURM_JOB_ID'),
        scope='Once-opened photometric-first prospective cohorts. Modular transfer, not full-catalog population refit. Reference and target use the same retained source/calibration realization.',
        pruning='Gaussian observation score bounded by retained mass and maximum Gaussian likelihood. PIT/CDF absolute error bounded by omitted response mass. Finite-prior support remains unverified.',
        omitted_log_mass=plan['discarded_response_log_mass'],outputs={x.name:sha(x) for x in out.iterdir() if x.is_file() and x.suffix in ('.npz','.csv')}))
    print('TRANSFER EVALUATION',a.cohort,n,flush=True)


if __name__=='__main__': main()

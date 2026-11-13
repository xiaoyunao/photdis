#!/usr/bin/env python3
"""Fit one registered luminosity candidate on allowed calibration sky only."""
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.luminosity import luminosity_transform

_DATA=_GRID=_PRIOR=_WEIGHTS=_SKY=_CELLS=_OUT=_INPUTS=None


def reference_curves(evidence, grid, calibration):
    result = np.empty((len(evidence['identity']),len(grid)))
    for i, channel in enumerate(evidence['fit_channel']):
        if channel == 0:
            log = norm.logpdf(evidence['p_corr'][i], 10**(2-.2*grid), evidence['gaia_parallax_error'][i])
        else:
            name = 'specdis' if channel == 1 else 'spdist'
            delta, scatter, fraction = calibration[channel-1]
            center = 10+5*np.log10(evidence[name+'_distance'][i])+delta
            error = evidence[name+'_sigma_mu'][i]
            log = np.logaddexp(np.log1p(-fraction)+norm.logpdf(grid, center, np.hypot(error,scatter)),
                np.log(fraction)+norm.logpdf(grid, center, np.hypot(error,2.)))
        # A per-star constant cannot affect the correction optimum. Retain it
        # separately in reporting if actual log predictive densities are needed.
        result[i] = np.exp(log-log.max())
    return result


def fit_draw(draw):
    path = _OUT/f'draw_{draw:02}.json'
    if path.exists():
        old = json.loads(path.read_text()); assert old['inputs'] == _INPUTS
        return old
    start = time.monotonic(); results = []; parameters = []
    for c in range(3):
        data = _DATA[c]; weight = data['weight'].copy()
        if draw: weight *= _SKY[draw-1, np.searchsorted(_CELLS, data['cell'])]
        weight *= len(weight)/weight.sum()
        def objective(x):
            likelihood = luminosity_transform(data['likelihood'], _GRID, *x)
            density = likelihood*_PRIOR
            denominator = density@_WEIGHTS
            score = np.einsum('nm,nm,m->n',density,data['reference'],_WEIGHTS)/denominator
            if not np.isfinite(score).all() or (score <= 0).any(): return 1e100
            return float(-weight@np.log(score)+.5*np.sum((x/.3)**2))
        alternatives = []
        for initial in (np.array([0.,0.]),np.array([0.,.3])):
            fit = minimize(objective,initial,method='L-BFGS-B',bounds=[(-1.,1.),(0.,1.)],
                options=dict(maxiter=200,ftol=1e-10,gtol=1e-5,finite_diff_rel_step=1e-4))
            alternatives.append(dict(initial=initial.tolist(),parameters=fit.x.tolist(),objective=float(fit.fun),
                success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),gradient=fit.jac.tolist()))
        valid = [x for x in alternatives if x['success'] and np.isfinite(x['objective']) and x['objective'] < 1e99]
        if not valid: raise RuntimeError('Luminosity optimizer failed for draw/class: '+str((draw,c,alternatives)))
        best = min(valid,key=lambda x:x['objective']); parameters.append(best['parameters'])
        results.append(dict(reference_class=c,alternatives=alternatives,selected=best,
            null_objective=objective(np.zeros(2)),n=len(weight),sky_cells=len(np.unique(data['cell'])),
            boundary=bool(abs(best['parameters'][0])>.999 or best['parameters'][1]>.999)))
    record = dict(inputs=_INPUTS,draw=draw,parameters=parameters,classes=results,
        elapsed_seconds=time.monotonic()-start,job=os.getenv('SLURM_JOB_ID'))
    save_json(path,record); print('LUMINOSITY FIT',draw,parameters,flush=True)
    return record


def main():
    global _DATA,_GRID,_PRIOR,_WEIGHTS,_SKY,_CELLS,_OUT,_INPUTS
    require_compute(); source = ROOT/'data/v3_1/luminosity_calibration'; _OUT = ROOT/'data/v3_1/luminosity_fit'; _OUT.mkdir(exist_ok=True)
    prepared = json.loads((source/'complete.json').read_text())
    paths = [source/'complete.json',source/'evidence.npz',ROOT/'data/v3_0/calibration.npz',
        ROOT/'scripts/v3_1_luminosity_fit.py',ROOT/'src/photdis/v3_1/luminosity.py',ROOT/'docs/v3_1_luminosity_correction_protocol.md']
    _INPUTS = {str(p.relative_to(ROOT)):sha(p) for p in paths}; freeze = _OUT/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == _INPUTS
    else: save_json(freeze,_INPUTS)
    if (_OUT/'complete.json').exists():
        done = json.loads((_OUT/'complete.json').read_text()); assert done['inputs'] == _INPUTS
        for name,digest in done['outputs'].items(): assert sha(_OUT/name) == digest
        return
    assert sha(source/'evidence.npz') == prepared['outputs']['evidence.npz']
    evidence = load(source/'evidence.npz'); use = evidence['partition'] == 0
    evidence = {k:v[use] for k,v in evidence.items()}; assert (evidence['TYPE'] == 'PSF').all()
    _GRID = np.linspace(-20,50,3501); _WEIGHTS = trapezoid_weights(_GRID); _PRIOR = norm.pdf(_GRID,15,5)
    likelihood = []
    for identity,c in zip(evidence['identity'],evidence['class_label']):
        path = source/f'object_{int(identity)}.json'; record = json.loads(path.read_text())
        assert sha(path) == prepared['objects'][path.name] and sha(path.with_suffix('.npz')) == record['sha256']
        value = load(path.with_suffix('.npz')); np.testing.assert_array_equal(value['coarse_grid'],_GRID)
        log = value['coarse_class_loglike'][c]; likelihood.append(np.exp(log-log.max()))
    likelihood = np.asarray(likelihood); reference = reference_curves(evidence,_GRID,load(paths[2])['map_parameters'])
    _CELLS = np.unique(evidence['cell']); rng = np.random.default_rng(318103)
    _SKY = rng.dirichlet(np.ones(len(_CELLS)),size=32)*len(_CELLS)
    atomic_npz(_OUT/'shared_sky.npz',cells=_CELLS,weights=_SKY,draw_ids=np.arange(1,33),
        identity=evidence['identity'],TYPE=evidence['TYPE'])
    _DATA = {}
    for c in range(3):
        selected = evidence['class_label'] == c
        assert selected.sum() >= 20 and len(np.unique(evidence['cell'][selected])) >= 2, 'Insufficient fitting support; retain failure'
        _DATA[c] = dict(likelihood=likelihood[selected],reference=reference[selected],
            weight=1/evidence['inclusion_probability'][selected],cell=evidence['cell'][selected])
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:
        results = list(pool.map(fit_draw,range(33)))
    parameters = np.array([r['parameters'] for r in results])
    atomic_npz(_OUT/'parameters.npz',point=parameters[0],bootstrap=parameters[1:],draw_ids=np.arange(1,33))
    rows = []
    for c in range(3):
        for j,name in enumerate(('delta_mu','sigma_mu')):
            lo,med,hi = np.quantile(parameters[1:,c,j],[.025,.5,.975])
            rows.append(dict(reference_class=c,parameter=name,point=float(parameters[0,c,j]),lower95=lo,median=med,upper95=hi))
    csv_rows(_OUT/'parameters.csv',rows)
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,outputs={p.name:sha(p) for p in _OUT.iterdir() if p.name != 'complete.json' and p.name != 'inputs.json'},
        job=os.getenv('SLURM_JOB_ID'),draws=32,classes=3,parameters=6,fit_rows=len(evidence['identity']),
        boundary_fits=sum(x['boundary'] for r in results for x in r['classes']),guard_evaluated=False,adopted=False,
        scope='Six-parameter conditional likelihood calibration on role3 fit-sky only. Shared sky bootstrap sensitivity, not exact posterior draws or independent absolute-scale proof. Numerical fine audit and photometry-only guards pending.'))


if __name__ == '__main__': main()

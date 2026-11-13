#!/usr/bin/env python3
"""Converged fixed-family SED refits for an independently regenerated library."""
import argparse
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json
from photdis.stage1 import require_compute
from photdis.v3.uncertainty import reference_quadrature_components, reference_logratio, one_step_ensemble


def main():
    require_compute(); p = argparse.ArgumentParser(); p.add_argument('--draw',type=int,required=True); a = p.parse_args()
    assert 0 <= a.draw < 32; started = time.monotonic(); old = ROOT/'data/v3_0'; root = ROOT/'data/v3_1/response_library'
    out = root/f'refits/draw_{a.draw+1000}'; out.mkdir(parents=True,exist_ok=True)
    paths = [old/'sed_fit.joblib',old/'sed_cloud.npz',old/'references.npz',root/'source_calibration.npz',
        root/'sed_case_weights.npz',root/'complete.json',ROOT/'scripts/v3_1_response_refit.py',
        ROOT/'src/photdis/v3/uncertainty.py',ROOT/'docs/v3_1_response_protocol.md']
    done = json.loads(paths[5].read_text())
    for path in paths[3:5]: assert sha(path) == done['outputs'][path.name]
    inputs = {str(path.relative_to(ROOT)):sha(path) for path in paths}; inputs['draw'] = a.draw+1000
    freeze = out/'inputs.json'; marker = out/'complete.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze,inputs)
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == inputs and sha(out/'model.joblib') == record['sha256']
        if not record['converged']: raise RuntimeError('Unresolved bounded refit; preserve flag and require explicit numerical amendment')
        return
    model = joblib.load(paths[0]); cloud = load(paths[1]); cal = load(paths[3]); weights = load(paths[4])
    assert model.converged and (cloud['TYPE'] == 'PSF').all() and (weights['TYPE'] == 'PSF').all()
    np.testing.assert_array_equal(model.training_identity,cloud['identity']); np.testing.assert_array_equal(weights['identity'],cloud['identity'])
    rows = cloud['rows']; channel = cloud['channel']; n = len(rows)
    reported = np.zeros(n); sigma = np.zeros(n)
    with np.load(paths[2]) as z:
        assert (z['TYPE'][rows] == 'PSF').all() and (z['role'][rows] == 1).all()
        for c,name in ((1,'specdis'),(2,'spdist')):
            use = channel == c; reported[use] = 10+5*np.log10(z[name+'_distance'][rows][use]); sigma[use] = z[name+'_sigma_mu'][rows][use]
    broad = reference_quadrature_components(cloud['reference_mu'],channel,reported,sigma,cal['map_parameters'])
    ratio,normalization = reference_logratio(cloud['reference_mu'],channel,reported,sigma,cal['map_parameters'],
        cal['parameters'][a.draw:a.draw+1],cloud['reference_weight'],broad)
    if np.quantile(abs(normalization),.99) > .01: raise RuntimeError('New source-calibration quadrature unresolved')
    case = weights['case_weights'][a.draw:a.draw+1]; del weights
    history = []; checkpoints = sorted(out.glob('iteration_*.json'))
    if checkpoints:
        record = json.loads(checkpoints[-1].read_text()); assert record['inputs'] == inputs
        path = checkpoints[-1].with_suffix('.joblib'); assert sha(path) == record['sha256']
        model = joblib.load(path); history = record['history']
    converged = False
    for iteration in range(len(history),64):
        models,norm = one_step_ensemble(model,cloud,ratio,case,workers=16)
        value = float(case[0]@norm[0]/case.sum()); history.append(value)
        if iteration >= 3 and abs(history[-1]-history[-2]) < 1e-4*(1+abs(value)): converged = True
        else: model = models[0]
        model.global_draw_id = a.draw+1000
        model.uncertainty_method = 'independently regenerated source-calibration and shared sky/source bootstrap, converged fixed-family EM'
        path = out/f'iteration_{iteration+1:04}.joblib'; joblib.dump(model,path.with_suffix('.tmp')); path.with_suffix('.tmp').replace(path)
        save_json(path.with_suffix('.json'),dict(inputs=inputs,sha256=sha(path),history=history,converged=converged))
        print('NEW SHARED REFIT',a.draw+1000,iteration+1,value,converged,time.monotonic()-started,flush=True)
        if converged: break
    model.converged = converged; joblib.dump(model,out/'model.tmp'); (out/'model.tmp').replace(out/'model.joblib')
    save_json(marker,dict(inputs=inputs,sha256=sha(out/'model.joblib'),history=history,converged=converged,
        global_draw_id=a.draw+1000,job=os.getenv('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-started,
        reference_quadrature_q99=float(np.quantile(abs(normalization),.99)),
        scope='All pre-registered responses retained. Independent Monte Carlo library conditional on same training data/family; target evidence not read.'))
    if not converged: raise RuntimeError('Bounded64iteration fit unresolved; preserve original flag')


if __name__ == '__main__': main()

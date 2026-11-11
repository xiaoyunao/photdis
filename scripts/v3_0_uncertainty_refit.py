#!/usr/bin/env python3
"""Converged fixed-family shared-bootstrap sensitivity to the one-update ensemble."""
import argparse
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json
from photdis.stage1 import require_compute
from photdis.v3.uncertainty import (reference_quadrature_components, reference_logratio,
    shared_case_weights, one_step_ensemble)
D = ROOT/'data/v3_0'


def main():
    require_compute(); parser = argparse.ArgumentParser()
    parser.add_argument('--draw', type=int, default=int(os.getenv('SLURM_ARRAY_TASK_ID', 0)))
    parser.add_argument('--max-iterations', type=int, default=64)
    args = parser.parse_args(); start = time.monotonic()
    paths = [D/'sed_fit.joblib', D/'sed_cloud.npz', D/'calibration.npz', D/'references.npz',
             ROOT/'src/photdis/v3/uncertainty.py', ROOT/'scripts/v3_0_uncertainty_refit.py']
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}; inputs.update(draw=args.draw, max_iterations=args.max_iterations)
    output = D/'uncertainty_refit'/f'draw_{args.draw:02}'; output.mkdir(parents=True, exist_ok=True)
    freeze = output/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze, inputs)
    marker = output/'complete.json'
    if marker.exists():
        old = json.loads(marker.read_text()); assert old['inputs'] == inputs
        assert old['sha256'] == sha(output/'model.joblib')
        if not old['converged']: raise RuntimeError('Shared refit did not converge; explicit numerical continuation needed')
        return
    model = joblib.load(paths[0]); cloud = load(paths[1]); cal = load(paths[2]); reference = load(paths[3])
    n = len(cloud['identity']); rows = cloud['rows']; channel = cloud['channel']
    assert (reference['TYPE'][rows] == 'PSF').all()
    np.testing.assert_array_equal(model.training_identity, cloud['identity'])
    reported_mu = np.zeros(n); sigma = np.zeros(n)
    for c, name in [(1, 'specdis'), (2, 'spdist')]:
        use = channel == c
        reported_mu[use] = 10+5*np.log10(reference[name+'_distance'][rows][use])
        sigma[use] = reference[name+'_sigma_mu'][rows][use]
    broad = reference_quadrature_components(cloud['reference_mu'], channel, reported_mu, sigma, cal['map_parameters'])
    all_weights, _, _ = shared_case_weights(cloud['identity'], reference['cell'][rows], draws=len(cal['parameters']))
    case = all_weights[args.draw:args.draw+1]
    ratio, normalization = reference_logratio(cloud['reference_mu'], channel, reported_mu, sigma,
        cal['map_parameters'], cal['parameters'][args.draw:args.draw+1], cloud['reference_weight'], broad)
    if np.quantile(abs(normalization), .99) > .01: raise RuntimeError('Reference quadrature failure')
    del reference, all_weights
    history = []; checkpoints = sorted(output.glob('iteration_*.json'))
    if checkpoints:
        record = json.loads(checkpoints[-1].read_text()); assert record['inputs'] == inputs
        path = checkpoints[-1].with_suffix('.joblib'); assert sha(path) == record['sha256']
        model = joblib.load(path); history = record['history']
    converged = False
    for iteration in range(len(history), args.max_iterations):
        models, norm = one_step_ensemble(model, cloud, ratio, case, workers=16)
        value = float(case[0]@norm[0]/case.sum())
        history.append(value)
        if iteration >= 3 and abs(history[-1]-history[-2]) < 1e-4*(1+abs(value)):
            # The likelihood belongs to the INPUT model, so retain that model on
            # convergence rather than silently labeling the unevaluated update.
            converged = True
        else: model = models[0]
        model.global_draw_id = args.draw
        model.uncertainty_method = 'converged weighted EM, shared sky/source Bayesian bootstrap and source-calibration draw'
        path = output/f'iteration_{iteration+1:04}.joblib'
        temp = path.with_suffix('.tmp'); joblib.dump(model, temp); temp.replace(path)
        save_json(path.with_suffix('.json'), dict(inputs=inputs, sha256=sha(path), history=history, converged=converged))
        print('SHARED REFIT', args.draw, iteration+1, value, converged, time.monotonic()-start, flush=True)
        if converged: break
    model.converged = converged
    joblib.dump(model, output/'model.tmp'); (output/'model.tmp').replace(output/'model.joblib')
    save_json(marker, dict(inputs=inputs, sha256=sha(output/'model.joblib'), history=history,
        converged=converged, global_draw_id=args.draw, elapsed_seconds=time.monotonic()-start,
        job=os.getenv('SLURM_JOB_ID'), method=model.uncertainty_method,
        limits='Fixed-family reference bootstrap sensitivity, not uncertainty over untested model families or independent absolute distance truth'))
    if not converged: raise RuntimeError('Shared weighted EM needs numerical continuation')


if __name__ == '__main__': main()

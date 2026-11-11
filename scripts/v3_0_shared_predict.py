#!/usr/bin/env python3
"""Common prediction path under a single globally shared reference/system draw."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.data import stable_rank
from photdis.v3.inference import infer_reference, reference_log_density
from v3_0_response import observation
import v3_0_predict as prediction
D = ROOT/'data/v3_0'
_CALIBRATION = None


def shared_observation(table, row):
    o = observation(table, row); o['shared_calibration_mag'] = _CALIBRATION[row]
    return o


def shared_infer(model, o, **kwargs):
    return infer_reference(model, o, calibration_mag=o['shared_calibration_mag'], **kwargs)


def shared_log_density(model, o, mu, **kwargs):
    return reference_log_density(model, o, mu, calibration_mag=o['shared_calibration_mag'], **kwargs)


def main():
    global _CALIBRATION
    require_compute(); parser = argparse.ArgumentParser()
    parser.add_argument('cohort', choices=['application', 'development', 'hyper', 'simulations'])
    parser.add_argument('--ensemble', choices=['influence', 'refit'], default='influence')
    parser.add_argument('--draw', type=int, default=int(os.getenv('SLURM_ARRAY_TASK_ID', 0)))
    parser.add_argument('--limit', type=int, default=0); parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args(); start = time.monotonic()
    source = D/'cohorts'/(args.cohort+'.npz'); shared_path = D/'uncertainty/shared_draws.npz'
    if args.ensemble == 'influence':
        model_path = D/'uncertainty/ensemble.joblib'; fitted = joblib.load(model_path)[args.draw]
    else:
        model_path = D/'uncertainty_refit'/f'draw_{args.draw:02}/model.joblib'
        record = json.loads(model_path.with_name('complete.json').read_text())
        assert record['converged'] and sha(model_path) == record['sha256']
        fitted = joblib.load(model_path)
    control_path = D/'controls/qrf_grz.joblib'
    t = load(source); shared = load(shared_path)
    if args.limit:
        rows = np.argsort(stable_rank(t['identity'], 302001))[:args.limit]
        t = {k: v[rows] for k, v in t.items()}
    assert (t['TYPE'] == 'PSF').all()
    _CALIBRATION, dust = calibration_vectors(t, shared, args.draw)
    paths = [source, shared_path, model_path, control_path, ROOT/'scripts/v3_0_shared_predict.py',
        ROOT/'scripts/v3_0_predict.py', ROOT/'scripts/v3_0_response.py']
    paths += [ROOT/'src/photdis/v3'/name for name in ('model.py', 'inference.py', 'fast.py', 'likelihood.py', 'compression.py', 'systematics.py', 'controls.py')]
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in paths}; frozen.update(draw=args.draw, limit=args.limit, ensemble=args.ensemble)
    output = D/'shared_predictions'/f'{args.cohort}_{args.ensemble}_{args.limit}'/f'draw_{args.draw:02}'
    output.mkdir(parents=True, exist_ok=True)
    freeze = output/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == frozen
    else: save_json(freeze, frozen)
    prediction._TABLE = t; prediction._MODEL = inference_model(fitted)
    prediction._CONTROL = joblib.load(control_path); prediction._FROZEN = frozen; prediction._DEST = output
    prediction._BANDS = np.ones(6, bool)
    prediction.observation = shared_observation; prediction.infer_reference = shared_infer
    prediction.reference_log_density = shared_log_density
    tasks = [np.arange(i, min(i+32, len(t['identity']))) for i in range(0, len(t['identity']), 32)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        markers = list(pool.map(prediction.chunk, tasks))
    atomic_npz(output/'shared_system.npz', identity=t['identity'], TYPE=t['TYPE'],
        global_draw_id=np.full(len(t['identity']), args.draw, 'i8'), calibration_mag=_CALIBRATION,
        dust_offset=dust, field=t['field'], cell=t['cell'])
    save_json(output/'complete.json', dict(inputs=frozen, rows=len(t['identity']), global_draw_id=args.draw,
        chunks={p: sha(p) for p in markers}, system_sha256=sha(output/'shared_system.npz'),
        system_scope='Declared target-relative-to-reference band and field-dust sensitivity priors; common reference+target zero points cancel.',
        elapsed_seconds=time.monotonic()-start, job=os.getenv('SLURM_JOB_ID')))
    print('SHARED PREDICTION', args.cohort, args.ensemble, args.draw, len(t['identity']), flush=True)


if __name__ == '__main__': main()

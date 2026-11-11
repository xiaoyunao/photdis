#!/usr/bin/env python3
"""Same-identity fixed QRF control; preserve its native histogram support."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, atomic_npz, save_json
from photdis.stage1 import require_compute
from photdis.baseline_v1 import integrate
from photdis.v3.controls import optical_magnitudes
from photdis.v3.data import stable_rank
D = ROOT/'data/v3_0'
_T = _MODEL = _OUTPUT = _INPUTS = None


def chunk(rows):
    path = _OUTPUT/f'chunk_{rows[0]:08}.npz'; marker = path.with_suffix('.json')
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == _INPUTS
        assert record['sha256'] == sha(path); return str(marker)
    start = time.monotonic(); mag, error = optical_magnitudes({k: v[rows] for k, v in _T.items()})
    # Historical Sobol helper has five band coordinates. Only native grz enter
    # this dim=2 control; unused WISE coordinates do not create observations.
    padded_mag = np.zeros((len(rows), 5)); padded_error = np.zeros_like(padded_mag)
    padded_mag[:, :3] = mag; padded_error[:, :3] = error
    out = integrate(_MODEL, padded_mag, padded_error, _T['identity'][rows], draws=128)['wide']
    atomic_npz(path, identity=_T['identity'][rows], TYPE=_T['TYPE'][rows], rows=rows,
        mu=out['mu'], Mg_edges=_MODEL.edges, g0=mag[:, 0], pdf=out['pdf'], supported=out['supported'],
        class_reference_contribution=out['weights'])
    save_json(marker, dict(inputs=_INPUTS, sha256=sha(path), rows=len(rows), seconds=time.monotonic()-start))
    return str(marker)


def main():
    global _T, _MODEL, _OUTPUT, _INPUTS
    require_compute(); parser = argparse.ArgumentParser()
    parser.add_argument('cohort', choices=['hyper', 'development', 'application'])
    parser.add_argument('--limit', type=int, default=0); parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    source = D/'cohorts'/(args.cohort+'.npz'); model = D/'controls/qrf_grz.joblib'
    _T = load(source)
    if args.limit:
        selected = np.argsort(stable_rank(_T['identity'], 302001))[:args.limit]
        _T = {k: v[selected] for k, v in _T.items()}
    assert (_T['TYPE'] == 'PSF').all()
    _MODEL = joblib.load(model)
    _INPUTS = {str(p.relative_to(ROOT)): sha(p) for p in [source, model, ROOT/'scripts/v3_0_qrf_predict.py',
        ROOT/'src/photdis/baseline_v0.py', ROOT/'src/photdis/baseline_v1.py', ROOT/'src/photdis/v3/controls.py']}
    _INPUTS.update(limit=args.limit, sobol_draws=128)
    _OUTPUT = D/'predictions'/f'{args.cohort}_qrf_{args.limit}'; _OUTPUT.mkdir(parents=True, exist_ok=True)
    freeze = _OUTPUT/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == _INPUTS
    else: save_json(freeze, _INPUTS)
    tasks = [np.arange(i, min(i+256, len(_T['identity']))) for i in range(0, len(_T['identity']), 256)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        paths = list(pool.map(chunk, tasks))
    save_json(_OUTPUT/'complete.json', dict(inputs=_INPUTS, rows=len(_T['identity']),
        chunks={p: sha(p) for p in paths}, job=os.getenv('SLURM_JOB_ID'),
        class_caveat='QRF leaf reference contribution includes unknown labels; not a joint physical class-distance posterior.'))
    print('QRF CONTROL COMPLETE', args.cohort, len(_T['identity']), flush=True)


if __name__ == '__main__': main()

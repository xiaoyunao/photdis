#!/usr/bin/env python3
"""Frozen-method validation/catalog predictions, identical native inference path."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from v3_0_validation_inputs import check_freeze
import v3_0_predict as common
D = ROOT/'data/v3_0'


def main():
    require_compute(); p = argparse.ArgumentParser()
    p.add_argument('scope', choices=['validation', 'catalog']); p.add_argument('name')
    p.add_argument('--workers', type=int, default=64); a = p.parse_args(); start = time.monotonic()
    method = check_freeze()
    if a.scope == 'validation':
        if a.name not in ('reservation', 'new_sky', 'independent_scale'): raise ValueError('Unknown frozen validation layer')
        folder = D/'validation'/a.name; source = folder/'photometry.npz'; meta_path = folder/'photometry.json'
        meta = json.loads(meta_path.read_text()); assert meta['method_freeze'] == method and meta['sha256'] == sha(source)
    else:
        field = int(a.name)
        if not 0 <= field < 14: raise ValueError('Unknown field')
        folder = D/'catalog'/f'field_{field:02}'; source = folder/'inputs.npz'; meta_path = folder/'prepared.json'
        meta = json.loads(meta_path.read_text()); assert meta['outputs'][source.name] == sha(source)
    model_path = D/'sed_fit.joblib'; control_path = D/'controls/qrf_grz.joblib'
    common._TABLE = load(source); fitted = joblib.load(model_path); assert fitted.converged
    assert (common._TABLE['TYPE'] == 'PSF').all()
    common._MODEL = inference_model(fitted); common._CONTROL = joblib.load(control_path); common._BANDS = np.ones(6, bool)
    common._DEST = folder/'predictions'; common._DEST.mkdir(exist_ok=True)
    paths = [source, meta_path, model_path, control_path, ROOT/'scripts/v3_0_candidate_predict.py', ROOT/'scripts/v3_0_predict.py', ROOT/'scripts/v3_0_response.py']
    paths += [ROOT/'src/photdis/v3'/x for x in ('inference.py', 'fast.py', 'likelihood.py', 'compression.py', 'model.py')]
    common._FROZEN = {str(x.relative_to(ROOT)): sha(x) for x in paths}; common._FROZEN['method_freeze'] = method
    freeze = common._DEST/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == common._FROZEN
    else: save_json(freeze, common._FROZEN)
    n = len(common._TABLE['identity']); tasks = [np.arange(i, min(i+32, n)) for i in range(0, n, 32)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('fork')) as pool:
        markers = []
        for path in pool.map(common.chunk, tasks):
            markers.append(path)
            if len(markers)%64 == 0: print('CANDIDATE PREDICT', a.scope, a.name, len(markers), len(tasks), flush=True)
    save_json(common._DEST/'complete.json', dict(inputs=common._FROZEN, rows=n, method_freeze=method,
        chunks={path: sha(path) for path in markers}, elapsed_seconds=time.monotonic()-start,
        status='Reference-prior candidate posterior; shared target product and external acceptance are separate', job=os.getenv('SLURM_JOB_ID')))
    print('CANDIDATE COMPLETE', a.scope, a.name, n, flush=True)


if __name__ == '__main__': main()

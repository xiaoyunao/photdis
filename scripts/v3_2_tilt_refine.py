#!/usr/bin/env python3
"""Refine only the entry contexts flagged by the first production pass."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os

import joblib
import numpy as np
from scipy.special import softmax

from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.target import sed_tilt_features
from photdis.v3_1.certified_support import CertifiedColorSupport
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_2.kernel_entry import kernel_entry
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_2_tilt_inputs import SHARDS, SOURCE, TILTS, OPTICAL, MEANS, SIGMAS, verified

LEVELS = ((2**22, 319711), (2**23, 319711), (2**23, 319712))
_STATE = None


def refine(row_check):
    table, cal, model, support, weights, markers = _STATE
    row = int(row_check['row'])
    identity = int(row_check['identity'])
    admission = row_check['admission']
    if int(table['identity'][row]) != identity or table['TYPE'][row] != 'PSF':
        raise ValueError(f'Refinement identity mismatch: {identity}')
    path = SOURCE / 'target_entry' / admission / f'object_{identity}' / 'complete.json'
    record = json.loads(verified(path, markers[str(path.relative_to(ROOT))]).read_text())
    if record['identity'] != identity or record['TYPE'] != 'PSF' or not record['passed']:
        raise ValueError(f'Invalid v3.1 entry source: {identity}')
    name = f"q{record['selected_draws']}_seed319711.npz"
    frozen = load(verified(path.parent / name, record['outputs'][name]))['alpha'][0]
    dusty = entry_model_with_local_dust(model, table['transmission'][row], table['EBV'][row])
    optical = (dusty, table['flux_error'][row, OPTICAL], table['transmission'][row, OPTICAL],
               cal['calibration_mag'][row, OPTICAL], MEANS, SIGMAS)
    estimates = np.stack([
        kernel_entry(*optical, support=support if admission == 'candidate' else None,
                     draws=draws, seed=seed)[0] for draws, seed in LEVELS])
    alpha = np.einsum('tkc,lkd->ltcd', weights, estimates)
    errors = np.stack([alpha[1] - alpha[0], alpha[1] - alpha[2]])
    zero_delta = float(abs(alpha[1, 0] - frozen).max())
    level_delta, seed_delta = [float(abs(error).max()) for error in errors]
    relative = max(float((abs(errors) / np.maximum(alpha[1], .001)).max()),
                   float((abs(alpha[1, 0] - frozen) / np.maximum(alpha[1, 0], .001)).max()))
    check = dict(row=row, identity=identity, TYPE='PSF', admission=admission,
                 v31_alpha_delta=zero_delta, max_tilt_level_delta=level_delta,
                 max_tilt_seed_delta=seed_delta, stabilized_relative_delta=relative,
                 passed=max(zero_delta, level_delta, seed_delta) < .0005 and relative < .01)
    return check, estimates


def main():
    global _STATE
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, choices=range(SHARDS), required=True)
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    name = f'shard_{args.shard:02}'
    parent = ROOT / 'data/v3_2/tilt_inputs'
    initial = parent / 'initial' / name
    base_path = initial / 'complete.json'
    base = json.loads(base_path.read_text())
    check_path = verified(initial / 'checks.csv', base['outputs']['checks.csv'])
    with check_path.open(newline='') as stream:
        flags = [r for r in csv.DictReader(stream) if r['passed'] == 'False']
    if len(flags) != base['failed']:
        raise ValueError('Initial flagged context count mismatch')
    for key in ('src/photdis/v3_2/kernel_entry.py', 'data/v3_1/target_matrices/source_markers.json'):
        verified(ROOT / key, base['inputs'][key])
    if flags:
        old = ROOT / 'data/v3_0'
        files = [old / 'cohorts/application.npz',
                 old / 'shared_predictions/application_refit_0/draw_12/shared_system.npz',
                 old / 'uncertainty_refit/draw_12/model.joblib', old / 'controls/qrf_grz.joblib']
        for p in files:
            verified(p, base['inputs'][str(p.relative_to(ROOT))])
        table, cal = map(load, files[:2])
        np.testing.assert_array_equal(table['identity'], cal['identity'])
        model = inference_model(joblib.load(files[2]))
        support = CertifiedColorSupport(joblib.load(files[3]))
        features, _ = sed_tilt_features(model)
        with np.errstate(divide='ignore'):
            log_prior = np.log(model.weights[:, None] * model.classes)
        weights = softmax(log_prior[None] + (TILTS @ features.T)[:, :, None], axis=1)
        markers = json.loads((SOURCE / 'target_matrices/source_markers.json').read_text())
        _STATE = table, cal, model, support, weights, markers

    output = parent / 'refined' / name
    output.mkdir(parents=True, exist_ok=False)
    checks, estimates = [], []
    if flags:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
            for check, entry in pool.map(refine, flags):
                checks.append(check)
                estimates.append(entry)
    arrays = dict(row=np.array([r['row'] for r in checks], dtype=int),
                  identity=np.array([r['identity'] for r in checks], dtype=np.int64),
                  TYPE=np.array([r['TYPE'] for r in checks], dtype='U3'),
                  admission=np.array([r['admission'] for r in checks], dtype='U9'),
                  passed=np.array([r['passed'] for r in checks], dtype=bool),
                  entry=np.stack(estimates) if estimates else np.empty((0, 3, 0, 5)))
    atomic_npz(output / 'patch.npz', **arrays)
    csv_rows(output / 'checks.csv', checks)
    save_json(output / 'complete.json', dict(
        shard=args.shard, all_PSF=True, source_marker_sha256=sha(base_path),
        levels=LEVELS, tilt_probes=TILTS.tolist(), initial_flags=len(flags),
        remaining_flags=sum(not r['passed'] for r in checks),
        code_sha256=sha(__file__), job=os.getenv('SLURM_JOB_ID'),
        outputs={p: sha(output / p) for p in ('patch.npz', 'checks.csv')}))


if __name__ == '__main__':
    main()

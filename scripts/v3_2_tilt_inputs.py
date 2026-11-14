#!/usr/bin/env python3
"""Build matched kernel likelihood and entry arrays for one frozen target shard."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import norm

from photdis.stage1 import require_compute
from photdis.v3.inference import DISTANCE_BASIS_MEAN as MEANS, DISTANCE_BASIS_SIGMA as SIGMAS
from photdis.v3.inference import kernel_loglike
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3.model import inference_model
from photdis.v3.selection import OPTICAL
from photdis.v3.target import sed_tilt_features
from photdis.v3_1.certified_support import CertifiedColorSupport
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_2.kernel_entry import kernel_entry
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_response import observation

SOURCE = ROOT / 'data/v3_1'
SHARDS = 32
LEVELS = ((2**20, 319711), (2**21, 319711), (2**21, 319712))
TILTS = np.array([[0., 0.], [.3, 0.], [-.3, 0.], [0., .3], [0., -.3]])
_STATE = None


def verified(path, digest):
    if sha(path) != digest:
        raise ValueError(f'Source hash mismatch: {path}')
    return path


def row_inputs(row):
    table, cal, roles, model, support, weights, markers = _STATE
    identity = int(table['identity'][row])
    if table['TYPE'][row] != 'PSF' or roles['TYPE'][row] != 'PSF':
        raise ValueError(f'Unverified native morphology: {identity}')

    native = SOURCE / 'target_native' / f'object_{identity}.npz'
    path = native.with_suffix('.json')
    record = json.loads(verified(path, markers[str(path.relative_to(ROOT))]).read_text())
    frozen = load(verified(native, record['sha256']))
    if record['identity'] != identity or not record['passed'] or frozen['TYPE'][0] != 'PSF':
        raise ValueError(f'Invalid frozen likelihood: {identity}')
    grid = frozen['coarse_grid']
    calibration = cal['calibration_mag'][row]
    logs, _ = kernel_loglike(model, observation(table, row), grid, calibration_mag=calibration)
    f = model.binary_reference_fraction
    native_log = logsumexp(logs + np.log([1 - f, f])[:, None, None], axis=0)
    radial = norm.logpdf(grid, MEANS[:, None], SIGMAS[:, None])
    log_scale = float(frozen['logscale'])
    component = np.exp(logsumexp(native_log[:, None, :] + radial[None, :, :] +
                                 np.log(trapezoid_weights(grid))[None, None, :], axis=2)
                       - log_scale)
    collapsed = weights[0].T @ component
    target = frozen['coarse_response'][0]
    relevant = np.maximum(collapsed, target) > np.exp(-30)
    native_delta = float(np.max(np.abs(np.log(collapsed[relevant]) - np.log(target[relevant]))))
    if not np.isfinite(native_delta) or native_delta > 1e-8:
        raise ValueError(f'Native likelihood identity failed: {identity} ({native_delta})')

    dusty = entry_model_with_local_dust(model, table['transmission'][row], table['EBV'][row])
    optical = (dusty, table['flux_error'][row, OPTICAL], table['transmission'][row, OPTICAL],
               calibration[OPTICAL], MEANS, SIGMAS)
    audits, entries = [], {}
    for admission in ('broad', 'candidate') if roles['candidate'][row] else ('broad',):
        pm = SOURCE / 'target_entry' / admission / f'object_{identity}' / 'complete.json'
        entry_record = json.loads(verified(pm, markers[str(pm.relative_to(ROOT))]).read_text())
        if not entry_record['passed'] or entry_record['identity'] != identity or entry_record['TYPE'] != 'PSF':
            raise ValueError(f'Invalid frozen entry: {identity} / {admission}')
        name = f"q{entry_record['selected_draws']}_seed319711.npz"
        exact = load(verified(pm.parent / name, entry_record['outputs'][name]))['alpha'][0]
        results = [kernel_entry(*optical, support=support if admission == 'candidate' else None,
                                draws=draws, seed=seed) for draws, seed in LEVELS]
        estimates = np.stack([a for a, _, _ in results])
        alpha = np.einsum('tkc,lkd->ltcd', weights, estimates)
        differences = np.stack([alpha[1] - alpha[0], alpha[1] - alpha[2]])
        v31_delta = float(np.max(abs(alpha[1, 0] - exact)))
        level_delta = float(np.max(abs(differences[0])))
        seed_delta = float(np.max(abs(differences[1])))
        rel_delta = max(float(np.max(abs(differences) / np.maximum(alpha[1], .001))),
                        float(np.max(abs(alpha[1, 0] - exact) / np.maximum(alpha[1, 0], .001))))
        passes = max(v31_delta, level_delta, seed_delta) < .0005 and rel_delta < .01
        audits.append(dict(row=row, identity=identity, TYPE='PSF', admission=admission,
                           native_log_delta=native_delta, v31_alpha_delta=v31_delta,
                           max_tilt_level_delta=level_delta, max_tilt_seed_delta=seed_delta,
                           stabilized_relative_delta=rel_delta, passed=passes))
        entries[admission] = estimates
    return row, component, log_scale, entries, audits


def main():
    global _STATE
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True, choices=range(SHARDS))
    parser.add_argument('--workers', type=int, choices=range(1, 33), default=20)
    parser.add_argument('--run', default='initial', choices=['initial'])
    args = parser.parse_args()
    old = ROOT / 'data/v3_0'
    pilot = SOURCE / 'target_native_pilot'
    matrix = SOURCE / 'target_matrices'
    source_paths = [pilot / 'roles.npz', old / 'cohorts/application.npz',
                    old / 'shared_predictions/application_refit_0/draw_12/shared_system.npz',
                    old / 'uncertainty_refit/draw_12/model.joblib', old / 'controls/qrf_grz.joblib']
    registered = ROOT / 'manifests/v3_2/tilt_pilot_refined/complete.json'
    refined = json.loads(registered.read_text())
    if not refined['passed']:
        raise ValueError('A passing refined input pilot is required')
    for p in source_paths:
        verified(p, refined['inputs'][str(p.relative_to(ROOT))])
    for key in ('src/photdis/v3_2/kernel_entry.py', 'src/photdis/v3/inference.py',
                'src/photdis/v3_1/selection_dust.py', 'src/photdis/v3/target.py'):
        verified(ROOT / key, refined['inputs'][key])
    verified(matrix / 'complete.json', sha(ROOT / 'manifests/v3_1/target_matrices/complete.json'))
    base = json.loads((matrix / 'complete.json').read_text())
    if not base['passed'] or not base['all_PSF']:
        raise ValueError('Frozen v3.1 arrays did not pass assembly')
    marker_path = verified(matrix / 'source_markers.json', base['outputs']['source_markers.json'])
    markers = json.loads(marker_path.read_text())

    roles, table, cal = map(load, source_paths[:3])
    np.testing.assert_array_equal(roles['identity'], table['identity'])
    np.testing.assert_array_equal(cal['identity'], table['identity'])
    if (len(table['identity']) != 26000 or int(roles['candidate'].sum()) != 3540 or
            not (table['TYPE'] == 'PSF').all() or not (roles['TYPE'] == 'PSF').all()):
        raise ValueError('Frozen broad/candidate PSF membership mismatch')
    model = inference_model(joblib.load(source_paths[3]))
    support = CertifiedColorSupport(joblib.load(source_paths[4]))
    features, _ = sed_tilt_features(model)
    with np.errstate(divide='ignore'):
        log_prior = np.log(model.weights[:, None] * model.classes)
    weights = softmax(log_prior[None] + (TILTS @ features.T)[:, :, None], axis=1)
    _STATE = table, cal, roles, model, support, weights, markers

    destination = ROOT / 'data/v3_2/tilt_inputs' / args.run / f'shard_{args.shard:02}'
    destination.mkdir(parents=True, exist_ok=False)
    rows = np.arange(args.shard, len(roles['identity']), SHARDS)
    responses, scales, broad, candidate, candidate_rows, checks = [], [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        for row, response, scale, entries, audits in pool.map(row_inputs, rows.tolist()):
            responses.append(response)
            scales.append(scale)
            broad.append(entries['broad'])
            if 'candidate' in entries:
                candidate.append(entries['candidate'])
                candidate_rows.append(row)
            checks.extend(audits)
            if len(responses) % 64 == 0:
                print(f'SHARD {args.shard} {len(responses)}/{len(rows)}', flush=True)
    candidate_entry = (np.stack(candidate) if candidate else
                       np.empty((0, len(LEVELS), len(model.weights), len(MEANS))))
    atomic_npz(destination / 'matrices.npz', row=rows, identity=roles['identity'][rows],
               TYPE=roles['TYPE'][rows], field=roles['field'][rows],
               design_weight=roles['design_weight'][rows], response=np.stack(responses),
               log_scale=np.asarray(scales), broad_entry=np.stack(broad),
               candidate_rows=np.asarray(candidate_rows, dtype=int), candidate_entry=candidate_entry)
    csv_rows(destination / 'checks.csv', checks)
    inputs = [*source_paths, registered, matrix / 'complete.json', marker_path,
              Path(__file__).resolve(), ROOT / 'src/photdis/v3_2/kernel_entry.py']
    save_json(destination / 'complete.json', dict(
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        outputs={p: sha(destination / p) for p in ('matrices.npz', 'checks.csv')},
        shard=args.shard, shards=SHARDS, n=len(rows), candidate_n=len(candidate_rows),
        all_PSF=True, non_PSF_excluded=0, missing_TYPE_excluded=0,
        contexts=len(checks), failed=sum(not r['passed'] for r in checks),
        levels=LEVELS, tilts=TILTS.tolist(), job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()

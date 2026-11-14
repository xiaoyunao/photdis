#!/usr/bin/env python3
"""Fit one matched population arm and compare selection integration levels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax

from photdis.stage1 import require_compute
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.model import inference_model
from photdis.v3.sampling import chain_diagnostics, hmc
from photdis.v3.target import sed_tilt_features
from photdis.v3_1.fixed_population import FixedKernelFieldPopulation
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_2_tilt_inputs import SHARDS, LEVELS, SOURCE, verified

_HIERARCHY = _FIT = None
DRAWS = 512
WARMUP = 512


def sample(index):
    return hmc(_HIERARCHY.objective, _FIT['map'], _FIT['metric'],
               draws=DRAWS, warmup=WARMUP, seed=319901 + index, target_acceptance=.9)


def assemble(case, root, expected_roles, *, refined):
    columns = {}
    markers = {}
    flags = dict(broad=0, candidate=0)
    source_inputs = None
    for shard in range(SHARDS):
        folder = root / f'shard_{shard:02}'
        path = folder / 'complete.json'
        record = json.loads(path.read_text())
        if (record['shard'] != shard or record['shards'] != SHARDS or not record['all_PSF']
                or record['levels'] != [list(x) for x in LEVELS]):
            raise ValueError(f'Wrong shard or quadrature identity: {path}')
        if source_inputs is None:
            source_inputs = record['inputs']
        elif record['inputs'] != source_inputs:
            raise ValueError(f'Target sources vary across shards: {path}')
        markers[str(path.relative_to(ROOT))] = sha(path)
        check_path = verified(folder / 'checks.csv', record['outputs']['checks.csv'])
        with check_path.open(newline='') as stream:
            checks = list(csv.DictReader(stream))
        for admission in flags:
            flags[admission] += sum(r['admission'] == admission and r['passed'] == 'False' for r in checks)
        data = load(verified(folder / 'matrices.npz', record['outputs']['matrices.npz']))
        np.testing.assert_array_equal(data['row'], np.arange(shard, len(expected_roles['identity']), SHARDS))
        matches = expected_roles['candidate'][data['row']]
        np.testing.assert_array_equal(data['candidate_rows'], data['row'][matches])
        if (len(matches) != record['n'] or int(matches.sum()) != record['candidate_n']
                or not (data['TYPE'] == 'PSF').all()):
            raise ValueError(f'Incomplete shard: {path}')
        for name in ('identity', 'TYPE', 'field', 'design_weight'):
            np.testing.assert_array_equal(data[name], expected_roles[name][data['row']])
        use = matches if case == 'candidate' else np.ones(len(matches), dtype=bool)
        for name in ('row', 'identity', 'TYPE', 'field', 'design_weight', 'response', 'log_scale'):
            columns.setdefault(name, []).append(data[name][use])
        entry = data[case + '_entry']
        shard_flags = {a: {int(r['row']) for r in checks if r['admission'] == a and
                            r['passed'] == 'False'} for a in flags}
        if sum(map(len, shard_flags.values())) != record['failed']:
            raise ValueError(f'Flag count mismatch: {path}')
        if refined:
            patch_folder = root.parent / 'refined' / folder.name
            patch_path = patch_folder / 'complete.json'
            patch_record = json.loads(patch_path.read_text())
            if patch_record['source_marker_sha256'] != markers[str(path.relative_to(ROOT))]:
                raise ValueError(f'Refinement source mismatch: {patch_path}')
            if patch_record['initial_flags'] != record['failed']:
                raise ValueError(f'Refinement count mismatch: {patch_path}')
            patch = load(verified(patch_folder / 'patch.npz', patch_record['outputs']['patch.npz']))
            markers[str(patch_path.relative_to(ROOT))] = sha(patch_path)
            if not (patch['TYPE'] == 'PSF').all():
                raise ValueError(f'Refinement morphology mismatch: {patch_path}')
            np.testing.assert_array_equal(patch['identity'], expected_roles['identity'][patch['row']])
            for admission, unresolved in shard_flags.items():
                selected = np.flatnonzero(patch['admission'] == admission)
                np.testing.assert_array_equal(np.sort(patch['row'][selected]), sorted(unresolved))
                if admission == case:
                    idx = np.searchsorted(data['row'][use], patch['row'][selected])
                    np.testing.assert_array_equal(data['row'][use][idx], patch['row'][selected])
                    if len(idx):
                        entry[idx] = patch['entry'][selected]
                flags[admission] += int(np.count_nonzero(~patch['passed'][selected]))
        else:
            for admission, unresolved in shard_flags.items():
                flags[admission] += len(unresolved)
        columns.setdefault('entry', []).append(entry)
    result = {name: np.concatenate(parts) for name, parts in columns.items()}
    order = np.argsort(result['row'])
    result = {name: array[order] for name, array in result.items()}
    expected = (np.flatnonzero(expected_roles['candidate']) if case == 'candidate' else
                np.arange(len(expected_roles['identity'])))
    np.testing.assert_array_equal(result['row'], expected)
    return result, markers, flags, source_inputs


def construct(data, model, mode, level):
    response = data['response']
    entry = data['entry'][:, level]
    fields, weights = data['field'], data['design_weight']
    if mode == 'free':
        return FastFieldPopulation(response, entry, model, fields, weights)
    prior = model.weights[:, None] * model.classes
    frequency = prior.sum(axis=0)
    prior /= frequency
    return FixedKernelFieldPopulation(
        np.einsum('nkd,kc->ncd', response, prior),
        np.einsum('nkd,kc->ncd', entry, prior), frequency, fields, weights)


def decode(hierarchy, parameters, mode):
    if mode == 'free':
        theta, beta, _, logits = hierarchy.decode(parameters)
    else:
        theta, logits = hierarchy.decode(parameters)
        beta = np.zeros(2)
    return theta, beta, logits


def selected_quantities(data, model, theta, beta, level):
    feature, _ = sed_tilt_features(model)
    with np.errstate(divide='ignore'):
        prior = softmax(np.log(model.weights[:, None] * model.classes)
                        + (feature @ beta)[:, None], axis=0)
    fields, owners = np.unique(data['field'], return_inverse=True)
    joint = np.empty((len(owners), 3, 5))
    entry = np.empty(len(owners))
    for i in range(len(fields)):
        use = owners == i
        values = np.einsum('nkd,kc->ncd', data['response'][use], prior) * theta[i]
        joint[use] = values / values.sum(axis=(1, 2), keepdims=True)
        entry[use] = np.einsum('nkd,kc,cd->n', data['entry'][use, level], prior, theta[i])
    return joint, entry


def main():
    global _HIERARCHY, _FIT
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['broad', 'candidate'])
    parser.add_argument('mode', choices=['zero', 'free'])
    parser.add_argument('--initial-only', action='store_true')
    parser.add_argument('--diagnostic-with-flags', action='store_true',
                        help='Permit a labeled sensitivity fit with unresolved entry contexts')
    args = parser.parse_args()
    root = ROOT / 'data/v3_2/tilt_inputs/initial'
    roles_path = SOURCE / 'target_native_pilot/roles.npz'
    pilot = ROOT / 'manifests/v3_2/tilt_pilot_refined/complete.json'
    model_path = ROOT / 'data/v3_0/uncertainty_refit/draw_12/model.joblib'
    sources = json.loads(pilot.read_text())['inputs']
    roles = load(verified(roles_path, sources[str(roles_path.relative_to(ROOT))]))
    model = inference_model(joblib.load(verified(model_path, sources[str(model_path.relative_to(ROOT))])))
    data, markers, flags, shard_inputs = assemble(args.case, root, roles, refined=not args.initial_only)

    if flags[args.case] and not args.diagnostic_with_flags:
        raise ValueError(f'{flags[args.case]} {args.case} contexts need refinement')
    run = 'diagnostic' if args.diagnostic_with_flags else 'initial'
    output = ROOT / 'data/v3_2/tilt_fits' / run / f'{args.case}_{args.mode}'
    output.mkdir(parents=True, exist_ok=False)
    fitted = {}
    for level in (0, 2, 1):
        hierarchy = construct(data, model, args.mode, level)
        fit = hierarchy.fit()
        fitted[level] = decode(hierarchy, fit['map'], args.mode)
        print(f'MAP {args.case}/{args.mode} level={level} gradient={fit["gradient_max"]}', flush=True)
        if level == 1:
            _HIERARCHY, _FIT = hierarchy, fit
        else:
            del hierarchy

    base_joint, base_entry = selected_quantities(data, model, *fitted[1][:2], 1)
    weights = data['design_weight'] / data['design_weight'].mean()
    comparisons = []
    for level in (0, 2):
        joint, entry = selected_quantities(data, model, *fitted[level][:2], level)
        _, fixed_entry = selected_quantities(data, model, *fitted[1][:2], level)
        comparisons.append(dict(
            level=level, maximum_tilt_delta=float(abs(fitted[1][1] - fitted[level][1]).max()),
            maximum_joint_basis_delta=float(abs(joint - base_joint).max()),
            maximum_expected_basis_count_delta=float(abs((joint - base_joint).sum(axis=0)).max()),
            maximum_fixed_parameter_entry_delta=float(abs(fixed_entry - base_entry).max()),
            total_fixed_parameter_log_entry_delta=float(weights @ np.log(fixed_entry / base_entry))))
    csv_rows(output / 'entry_sensitivity.csv', comparisons)
    atomic_npz(output / 'map.npz', theta=np.stack([fitted[i][0] for i in range(3)]),
               beta=np.stack([fitted[i][1] for i in range(3)]), fields=_HIERARCHY.fields)

    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('fork')) as pool:
        chains = list(pool.map(sample, range(4)))
    draws = np.array([c['samples'] for c in chains])
    checks = chain_diagnostics(draws)
    decoded = [decode(_HIERARCHY, v, args.mode) for v in draws.transpose(1, 0, 2).reshape(-1, draws.shape[-1])]
    atomic_npz(output / 'posterior.npz', theta=np.stack([x[0] for x in decoded]),
               beta=np.stack([x[1] for x in decoded]), global_logits=np.stack([x[2] for x in decoded]),
               fields=_HIERARCHY.fields, identity=data['identity'], TYPE=data['TYPE'], chains=draws,
               rhat=checks['rhat'], ess=checks['ess'], acceptance=np.stack([c['acceptance'] for c in chains]))
    divergences = sum(c['divergent'] for c in chains)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (
        roles_path, pilot, model_path, Path(__file__).resolve(),
        ROOT / 'src/photdis/v3/fast_hierarchy.py', ROOT / 'src/photdis/v3/hierarchy.py',
        ROOT / 'src/photdis/v3/sampling.py', ROOT / 'src/photdis/v3_1/fixed_population.py')}
    save_json(output / 'complete.json', dict(
        case=args.case, tilt=args.mode, n=len(data['identity']), all_PSF=True,
        outputs={p.name: sha(p) for p in sorted(output.iterdir())}, inputs=inputs,
        shard_markers=markers, shard_inputs=shard_inputs, retained_context_flags=flags,
        diagnostic_with_flags=args.diagnostic_with_flags,
        refined_inputs=not args.initial_only,
        job=os.getenv('SLURM_JOB_ID'), draws_per_chain=DRAWS, warmup=WARMUP,
        max_rhat=checks['max_rhat'], minimum_ess=checks['minimum_ess'], divergences=divergences,
        sampler_passed=checks['max_rhat'] < 1.05 and checks['minimum_ess'] >= 100 and divergences == 0,
        scope='Conditional zero-correction population fit. Context flags and MAP sensitivity retained. '
              'External scoring, count and fitted-posterior selection sensitivity remain separate.'))


if __name__ == '__main__':
    main()

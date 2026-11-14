#!/usr/bin/env python3
"""Test both shared NGC288 quadrature axes with full and Gaussian likelihoods."""
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from photdis.stage1 import require_compute
from photdis.v3_1.structure import structure_quadrature
from photdis.v3_1.structure_posterior import common_structure_posterior
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_1_structure_infer import compare
from v3_2_tilt_inputs import verified

BRANCHES = ('full', 'gaussian')
COMPARISONS = (
    ('distance_first', 'm128_z1024', 'm64_z1024'),
    ('distance_final', 'm256_z1024', 'm128_z1024'),
    ('odds_final', 'm256_z1024', 'm256_z512'),
    ('joint_final', 'm256_z1024', 'm128_z512'),
    ('native_final', 'native_coarse_m256_z1024', 'm256_z1024'),
    ('entry_seed_final', 'entry_seed_m256_z1024', 'm256_z1024'),
    ('entry_level_final', 'entry_level_m256_z1024', 'm256_z1024'),
)


def evaluate(old, nodes, branch, distance, odds, *, native_grid=1, entry_variant=0):
    mu, lmu, zeta, lz = structure_quadrature(distance, odds)
    # Native sidecars preserve the original broad background for each state.
    background = logsumexp(old['logresponse'][native_grid, branch, :, :, :5], axis=-1)-np.log(5)
    background_entry = old['entry'][entry_variant, :, :, :5].mean(axis=-1)
    if distance == 64:
        available = old['common_mu']
        log_response = old['logresponse'][native_grid, branch, :, :, 5:]
        selection = old['entry'][entry_variant, :, :, 5:]
    else:
        available = nodes['mu']
        log_response = nodes['response'][native_grid, branch]
        selection = nodes['entry'][entry_variant]
    ix = np.searchsorted(available, mu)
    np.testing.assert_array_equal(available[ix], mu)
    # Permit only roundoff at the boundary, already checked before assembly.
    return common_structure_posterior(
        log_response[..., ix], background, np.clip(selection[..., ix], 0, 1), background_entry,
        old['radius_deg'], mu, lmu, zeta, lz)


def validate_posterior(result, n):
    if not np.isfinite(result['log_evidence']):
        raise ValueError('Finite joint log evidence required')
    for key in ('global_weights', 'response_weights', 'common_mu_probability', 'zeta_probability'):
        weight = result[key]
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError(f'Invalid probability: {key}')
        np.testing.assert_allclose(weight.sum(), 1, rtol=0, atol=1e-10)
    members = result['per_object_member_probability']
    if members.shape != (n,) or (members < -1e-12).any() or (members > 1+1e-12).any():
        raise ValueError('Invalid object membership probabilities')
    np.testing.assert_allclose(result['mean'], [members.sum(), n-members.sum()], rtol=0, atol=1e-8)
    cov = result['covariance']
    np.testing.assert_allclose(cov.sum(axis=0), 0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(cov, result['conditional_covariance']+result['shared_covariance'], rtol=0, atol=1e-8)
    if np.linalg.eigvalsh(cov).min() < -1e-8:
        raise ValueError('Invalid shared count covariance')


def main():
    require_compute()
    root = ROOT/'data/v3_2'
    existing = ROOT/'data/v3_1/structure/matrices'
    marker = verified(existing/'complete.json', sha(ROOT/'manifests/v3_1/structure/matrices/complete.json'))
    record = json.loads(marker.read_text())
    table = load(verified(existing/'matrices.npz', record['outputs']['matrices.npz']))
    ids, types, states = [table[k] for k in ('identity', 'TYPE', 'response_ids')]
    if (not record['passed'] or not record['all_PSF'] or len(ids) != 512
            or len(np.unique(ids)) != 512 or not (types == 'PSF').all()):
        raise ValueError('Complete frozen parent sample required')
    np.testing.assert_array_equal(states, np.arange(32))
    sources = [marker, Path(__file__).resolve(), ROOT/'docs/v3_2_structure_protocol.md',
               ROOT/'src/photdis/v3_1/structure.py', ROOT/'src/photdis/v3_1/structure_posterior.py',
               ROOT/'scripts/v3_1_structure_infer.py']
    means = np.unique(np.r_[structure_quadrature(128, 64)[0], structure_quadrature(256, 64)[0]])
    entries, responses, input_failures = [], [], []
    for draw in states:
        folder = root/'structure_distance'/f'draw_{draw:02}'
        p = folder/'complete.json'
        state = json.loads(p.read_text())
        if (state.get('pilot') is not False or not state['all_PSF'] or state['draw'] != draw or state['n'] != len(ids)
                or state['inputs'][str(marker.relative_to(ROOT))] != sha(marker)):
            raise ValueError(f'Mismatched distance-node receipt: {draw}')
        a = load(verified(folder/'nodes.npz', state['outputs']['nodes.npz']))
        for key, expected in [('mu', means), ('identity', ids), ('TYPE', types), ('row', np.arange(len(ids)))]:
            np.testing.assert_array_equal(a[key], expected)
        response, entry = a['response'], a['entry']
        if (response.shape != (len(ids), 2, 2, len(means)) or entry.shape != (len(ids), 3, len(means))
                or not np.isfinite(response).all() or not np.isfinite(entry).all()
                or entry.min() < -1e-12 or entry.max() > 1+1e-12):
            raise ValueError(f'Invalid distance-node payload: {draw}')
        with verified(folder/'checks.csv', state['outputs']['checks.csv']).open(newline='') as stream:
            detail = list(csv.DictReader(stream))
        kinds = ('full_native', 'gaussian_native', 'entry_second_seed',
                 'entry_previous_level', 'frozen_entry_closure')
        expected = {(str(i), str(identity), 'PSF', kind)
                    for i, identity in enumerate(ids) for kind in kinds}
        actual = {(r['row'], r['identity'], r['TYPE'], r['check']) for r in detail}
        failures = sum(r['passed'] != 'True' for r in detail)
        if (actual != expected or len(detail) != len(expected)
                or any(r['passed'] not in ('True', 'False') for r in detail)
                or failures != state['failures'] or (failures == 0) != state['passed']):
            raise ValueError(f'Distance-node check inventory mismatch: {draw}')
        sources.append(p)
        input_failures.append(dict(draw=int(draw), context_checks_failed=int(state['failures']), passed=bool(state['passed'])))
        responses.append(response)
        entries.append(entry)
    nodes = dict(mu=means, response=np.asarray(responses).transpose(2, 3, 0, 1, 4),
                 entry=np.asarray(entries).transpose(2, 0, 1, 3))
    odds = root/'structure_odds_audit'
    odds_marker = odds/'complete.json'
    odds_record = json.loads(odds_marker.read_text())
    if not odds_record['all_PSF'] or odds_record['n'] != 512 or odds_record['responses'] != 32:
        raise ValueError('Complete fixed-distance odds audit required')
    if odds_record['inputs'][str(marker.relative_to(ROOT))] != sha(marker):
        raise ValueError('Odds audit binds a different frozen parent')
    sources.append(odds_marker)
    out = root/'structure_joint'
    out.mkdir(parents=True, exist_ok=False)
    summaries, checks, final = [], [], []
    for branch, name in enumerate(BRANCHES):
        results = {}
        key = f'{name}_1024.npz'
        results['m64_z1024'] = load(verified(odds/key, odds_record['outputs'][key]))
        for k, expected in [('identity', ids), ('TYPE', types), ('response_ids', states)]:
            np.testing.assert_array_equal(results['m64_z1024'][k], expected)
        for key, m, z, g, e in (
            ('m128_z512', 128, 512, 1, 0), ('m128_z1024', 128, 1024, 1, 0),
            ('m256_z512', 256, 512, 1, 0), ('m256_z1024', 256, 1024, 1, 0),
            ('native_coarse_m256_z1024', 256, 1024, 0, 0),
            ('entry_seed_m256_z1024', 256, 1024, 1, 1),
            ('entry_level_m256_z1024', 256, 1024, 1, 2),
        ):
            result = evaluate(table, nodes, branch, m, z, native_grid=g, entry_variant=e)
            validate_posterior(result, len(ids))
            atomic_npz(out/f'{name}_{key}.npz', identity=ids, TYPE=types, response_ids=states, **result)
            results[key] = result
            print(name, key, float(result['mean'][0]), flush=True)
        final.append(results['m256_z1024'])
        for kind, variant, reference in COMPARISONS:
            checks.append(dict(branch=name, comparison=kind, variant=variant, reference=reference,
                               **compare(results[variant], results[reference])))
        for key, result in results.items():
            summaries.append(dict(branch=name, variant=key, log_evidence=float(result['log_evidence']),
                                  mean_members=float(result['mean'][0]), member_sd=float(np.sqrt(result['covariance'][0, 0])),
                                  response_ess=float(1/np.sum(result['response_weights']**2)),
                                  q025=float(result['common_mu_quantiles'][0]),
                                  q50=float(result['common_mu_quantiles'][2]),
                                  q975=float(result['common_mu_quantiles'][4])))
    delta = compare(*final)
    # Full versus Gaussian is a scientific contrast, not a convergence gate.
    delta.pop('passed')
    delta['mean_members_full_minus_gaussian'] = float(final[0]['mean'][0]-final[1]['mean'][0])
    save_json(out/'branch_comparison.json', delta)
    csv_rows(out/'summary.csv', summaries)
    csv_rows(out/'checks.csv', checks)
    csv_rows(out/'input_checks.csv', input_failures)
    inputs_passed = all(r['passed'] for r in input_failures)
    integration_passed = all(r['passed'] for r in checks if r['comparison'] != 'distance_first')
    save_json(out/'complete.json', dict(
        inputs={str(p.relative_to(ROOT)): sha(p) for p in sources},
        outputs={p.name: sha(p) for p in sorted(out.iterdir())}, job=os.getenv('SLURM_JOB_ID'),
        all_PSF=True, n=len(ids), responses=len(states), input_checks_passed=inputs_passed,
        integration_checks_passed=integration_passed, passed=inputs_passed and integration_passed,
        scope='Seen-system NGC288 quadrature. Fixed 32-state finite response set and original gates.'))


if __name__ == '__main__':
    main()

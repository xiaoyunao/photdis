#!/usr/bin/env python3
"""Verify the compact final NGC288 distance-by-odds results."""
import csv
import json
from math import isclose, isfinite
from pathlib import Path

from v3_2_structure_distance_audit import audit as distance_audit, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'manifests/v3_2/structure_joint'
BRANCHES = ('full', 'gaussian')
COMPARISONS = {
    'distance_first': ('m128_z1024', 'm64_z1024'),
    'distance_final': ('m256_z1024', 'm128_z1024'),
    'odds_final': ('m256_z1024', 'm256_z512'),
    'joint_final': ('m256_z1024', 'm128_z512'),
    'native_final': ('native_coarse_m256_z1024', 'm256_z1024'),
    'entry_seed_final': ('entry_seed_m256_z1024', 'm256_z1024'),
    'entry_level_final': ('entry_level_m256_z1024', 'm256_z1024'),
}
TOLERANCES = {
    'maximum_member_probability_delta': .003,
    'maximum_mean_count_delta': .5,
    'maximum_common_mu_quantile_delta': .03,
    'log_evidence_delta': .05,
}
LABELS = ('member', 'background')
COMPONENTS = ('total', 'conditional', 'shared')
VARIANTS = {name for pair in COMPARISONS.values() for name in pair}
QUANTILES = ('q025', 'q16', 'q50', 'q84', 'q975')


def read_table(path, keys, expected):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    index = {tuple(row[key] for key in keys): row for row in rows}
    if len(rows) != len(index) or index.keys() != expected:
        raise ValueError(f'Incomplete or duplicated table: {path.name}')
    return index


def numeric(row, fields):
    values = tuple(float(row[key]) for key in fields)
    if not all(map(isfinite, values)):
        raise ValueError(f'Nonfinite fields: {fields}')
    return values


def close(left, right):
    return isclose(left, right, abs_tol=1e-8, rel_tol=1e-10)


def quantile_shift(left, right):
    return max(abs(x-y) for x, y in zip(numeric(left, QUANTILES), numeric(right, QUANTILES)))


def audit_tables(folder, receipt, roster, state_failures):
    names = ('input_checks.csv', 'checks.csv', 'summary.csv', 'membership.csv',
             'count_covariance.csv', 'branch_comparison.json')
    expected = {*names, *(f'{b}_{v}.npz' for b in BRANCHES for v in VARIANTS if v != 'm64_z1024')}
    if receipt['outputs'].keys() != expected:
        raise ValueError('Incomplete joint output inventory')
    for name in names:
        if digest((folder / name).read_bytes()) != receipt['outputs'][name]:
            raise ValueError(f'Output checksum mismatch: {name}')
    states = read_table(folder/'input_checks.csv', ('draw',), {(str(i),) for i in range(32)})
    for (draw,), row in states.items():
        failures = state_failures[int(draw)]
        if int(row['context_checks_failed']) != failures or row['passed'] != str(failures == 0):
            raise ValueError(f'Input flag mismatch for state {draw}')

    summary = read_table(folder/'summary.csv', ('branch', 'variant'),
                         {(b, v) for b in BRANCHES for v in VARIANTS})
    for row in summary.values():
        _, members, sd, ess = numeric(row, ('log_evidence', 'mean_members', 'member_sd', 'response_ess'))
        quantiles = numeric(row, QUANTILES)
        if not (0 <= members <= len(roster) and sd >= 0 and 1-1e-10 <= ess <= 32+1e-10
                and list(quantiles) == sorted(quantiles)):
            raise ValueError('Invalid joint posterior summary')
    checks = read_table(folder/'checks.csv', ('branch', 'comparison'),
                        {(b, c) for b in BRANCHES for c in COMPARISONS})
    for (branch, comparison), row in checks.items():
        v, r = COMPARISONS[comparison]
        values = numeric(row, TOLERANCES)
        a, b = (summary[branch, variant] for variant in (v, r))
        passed = all(abs(x) < limit for x, limit in zip(values, TOLERANCES.values()))
        if ((row['variant'], row['reference']) != (v, r) or min(values[:3]) < 0
                or row['passed'] != str(passed)
                or not close(values[1], abs(float(a['mean_members']) - float(b['mean_members'])))
                or not close(values[2], quantile_shift(a, b))
                or not close(values[-1], float(a['log_evidence']) - float(b['log_evidence']))):
            raise ValueError(f'Invalid quadrature contrast: {branch}/{comparison}')

    membership = read_table(folder/'membership.csv', ('identity',), {(i,) for i in roster})
    probabilities = []
    for row in membership.values():
        full, gaussian, delta = numeric(row, ('full', 'gaussian', 'full_minus_gaussian'))
        if (row['TYPE'] != 'PSF' or min(full, gaussian) < -1e-12 or max(full, gaussian) > 1+1e-12
                or not close(full-gaussian, delta)):
            raise ValueError(f'Invalid membership row: {row["identity"]}')
        probabilities.append((full, gaussian))
    covariance = read_table(folder/'count_covariance.csv', ('branch', 'component', 'row', 'column'),
                            {(b, k, i, j) for b in BRANCHES for k in COMPONENTS for i in LABELS for j in LABELS})
    finals = [summary[branch, 'm256_z1024'] for branch in BRANCHES]
    for branch, final, values in zip(BRANCHES, finals, zip(*probabilities)):
        if not close(sum(values), float(final['mean_members'])):
            raise ValueError(f'Membership probabilities do not sum to the count: {branch}')
        matrix = {}
        for component in COMPONENTS:
            cells = [numeric(covariance[branch, component, i, j], ('value',))[0] for i in LABELS for j in LABELS]
            variance = cells[0]
            if variance < -1e-8 or not all(close(x, y) for x, y in zip(cells, (variance, -variance, -variance, variance))):
                raise ValueError(f'Nonconserving count covariance: {branch}/{component}')
            matrix[component] = variance
        if (not close(matrix['total'], float(final['member_sd'])**2)
                or not close(matrix['total'], matrix['conditional']+matrix['shared'])):
            raise ValueError(f'Count covariance decomposition fails: {branch}')

    delta = json.loads((folder/'branch_comparison.json').read_text())
    expected_delta = {
        'maximum_member_probability_delta': max(abs(a-b) for a, b in probabilities),
        'mean_members_full_minus_gaussian': float(finals[0]['mean_members']) - float(finals[1]['mean_members']),
        'maximum_common_mu_quantile_delta': quantile_shift(*finals),
        'log_evidence_delta': float(finals[0]['log_evidence']) - float(finals[1]['log_evidence']),
    }
    expected_delta['maximum_mean_count_delta'] = abs(expected_delta['mean_members_full_minus_gaussian'])
    if set(delta) != set(expected_delta) or not all(close(float(delta[k]), v) for k, v in expected_delta.items()):
        raise ValueError('Final likelihood contrast does not reconcile')
    input_passed = not any(state_failures.values())
    final_failed = [f'{b}/{c}' for (b, c), row in checks.items() if c != 'distance_first' and row['passed'] == 'False']
    if (receipt['input_checks_passed'] is not input_passed or receipt['integration_checks_passed'] is not (not final_failed)
            or receipt['passed'] is not (input_passed and not final_failed)):
        raise ValueError('Joint verdict does not match the component results')
    return dict(n=len(membership), n_states=len(states), n_cells=len(covariance),
                comparisons=len(checks), retained_input_failures=sum(state_failures.values()),
                final_failed=final_failed, passed=receipt['passed'])


def audit():
    preparatory = distance_audit(complete=True)
    raw = (BASE/'complete.json').read_bytes()
    receipt = json.loads(raw)
    if receipt['n'] != 512 or receipt['responses'] != 32 or receipt['all_PSF'] is not True:
        raise ValueError('Joint result has a different census')
    inputs = receipt['inputs']
    for row in preparatory['records']:
        path = f'data/v3_2/structure_distance/draw_{row["draw"]:02}/complete.json'
        if inputs[path] != row['receipt_sha256']:
            raise ValueError(f'Joint ancestor differs from the audited state: {path}')
    for source, expected in inputs.items():
        path = ROOT / source
        if source.startswith('data/'):
            path = ROOT / source.replace('data/', 'manifests/', 1)
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f'Unverified joint ancestor: {source}')
    with (BASE.parent/'structure_distance/draw_00/checks.csv').open(newline='') as stream:
        roster = {row['identity'] for row in csv.DictReader(stream)}
    state_failures = {r['draw']: sum(r['failures'].values()) for r in preparatory['records']}
    return dict(job=receipt['job'], receipt_sha256=digest(raw),
                **audit_tables(BASE, receipt, roster, state_failures))


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))

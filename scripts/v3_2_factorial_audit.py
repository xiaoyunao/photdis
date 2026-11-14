#!/usr/bin/env python3
"""Recompute the factorial scores and verify the separated numerical effects."""
import csv
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

from v3_2_opened_audit import check_stats, index, read_csv
from v3_2_tilt_contrasts import ARMS, contrast_rows, matched_rows
from v3_2_tilt_fit_audit import audit as fit_audit
from v3_2_tilt_report import historical_decomposition

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v3_2'
ROLES = {'new_sky_field': 167, 'new_source_seen_sky_field': 352}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_folder(folder, skipped=()):
    marker = folder/'complete.json'
    receipt = json.loads(marker.read_text())
    if receipt['all_PSF'] is not True or receipt['n'] != 526:
        raise ValueError(f'Expected full PSF forecast: {marker}')
    for name, expected in receipt['outputs'].items():
        if name not in skipped and digest(folder/name) != expected:
            raise ValueError(f'Changed source hash: {folder/name}')
    return receipt


def compare_table(path, result, keys):
    observed = index(read_csv(path), lambda row: tuple(str(row[key]) for key in keys))
    expected = index(result, lambda row: tuple(str(row[key]) for key in keys))
    if observed.keys() != expected.keys():
        raise ValueError(f'Missing contrast in {path}')
    for key, row in expected.items():
        check_stats(observed[key], row)
    return len(expected)


def audit():
    fits = fit_audit(complete=True)
    if fits['verified'] != 4 or not all(r['sampler_passed'] for r in fits['records']):
        raise ValueError('A registered population fit is incomplete or failed sampling')
    forecast = BASE/'tilt_predictions/initial'
    report = BASE/'tilt_report/initial'
    markers = (verify_folder(forecast), verify_folder(report, ('joint_covariance.npz',)))
    for name, expected in markers[1]['inputs'].items():
        path = ROOT/name.replace('data/v3_2/', 'manifests/v3_2/', 1)
        if digest(path) != expected:
            raise ValueError(f'Report ancestor differs: {name}')
    with (forecast/'objects.csv').open(newline='') as stream:
        objects = matched_rows(csv.DictReader(stream))
    if len(objects) != 526 or set(markers[0]['arms']) != set(ARMS):
        raise ValueError('Wrong factorial identity count or arms')
    anchors = index(read_csv(BASE/'opened_diagnostics/objects.csv'),
                    lambda row: (int(row['identity']), row['view']))
    contrasts = compare_table(report/'factorial.csv', contrast_rows(objects),
                              ('role', 'stratum', 'metric', 'contrast'))
    path_rows = compare_table(report/'historical_decomposition.csv',
                              historical_decomposition(objects, anchors),
                              ('role', 'stratum', 'metric', 'component'))

    keys = ('role', 'arm', 'level')
    checks = index(read_csv(report/'entry_sensitivity.csv'), lambda r: tuple(r[k] for k in keys))
    rows = read_csv(report/'entry_counts.csv')
    bins = index(rows, lambda r: tuple(r[k] for k in (*keys, 'class_index', 'distance_bin')))
    expected = {(role, arm, level) for role in ROLES for arm in ARMS for level in ('0', '2')}
    if checks.keys() != expected or bins.keys() != {
            (*k, str(c), str(b)) for k in expected for c in range(3) for b in range(6)}:
        raise ValueError('Incomplete entry comparison inventory')
    for key, row in checks.items():
        members = [r for k, r in bins.items() if k[:3] == key]
        base, alternative, shifts = (list(map(float, (r[field] for r in members)))
                                     for field in ('base_expected_count', 'expected_count', 'count_delta'))
        n = ROLES[key[0]]
        worst = max(map(abs, shifts))
        if (int(row['n']) != n or not all(isclose(sum(x), n, abs_tol=1e-8) for x in (base, alternative))
                or not all(isclose(a-b, s, abs_tol=1e-10) for b, a, s in zip(base, alternative, shifts))
                or not isclose(worst, float(row['maximum_mean_count_delta']), abs_tol=1e-10)
                or row['count_pass'] != str(worst < .5)
                or row['probability_pass'] != str(float(row['maximum_joint_probability_delta']) < .003)):
            raise ValueError(f'Entry sensitivity does not reconcile: {key}')

    transfers = [dict(arm=fit['arm'], **row) for fit in fits['records'] for row in fit['posterior_entry']]
    return dict(n=len(objects), factorial_contrasts=contrasts, historical_contrasts=path_rows,
                entry_count_bins=len(bins),
                max_abs_log_normalizer=max(abs(r['absolute_log_normalizer_delta']) for r in transfers),
                max_relative_log_weight_span=max(r['relative_log_weight_span'] for r in transfers),
                minimum_weight_ess=min(r['weight_ess'] for r in transfers),
                max_basis_count_delta=max(r['maximum_expected_basis_count_delta'] for r in transfers),
                max_opened_distance_bin_count_delta=max(float(r['maximum_mean_count_delta']) for r in checks.values()),
                scope='Compact output checks. Shared draw covariance stays bound by the allocated report receipt.')


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))

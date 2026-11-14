#!/usr/bin/env python3
"""Recompute the factorial scores and verify the separated numerical effects."""
import csv
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path

import numpy as np

from v3_2_opened_audit import check_stats, index, read_csv
from v3_2_tilt_contrasts import ARMS, METRICS, contrast_rows, matched_rows
from v3_2_tilt_fit_audit import audit as fit_audit
from v3_2_tilt_report import historical_decomposition
from v3_2_tilt_size_control import SEED, SIZE

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


def audit_size_control(objects):
    folder = BASE/'tilt_size_control'
    comparison = folder/'comparison/complete.json'
    paired = folder/'comparison/paired.csv'
    sample = folder/'sample.csv'
    record = json.loads(comparison.read_text())
    archived = json.loads((folder/'audit.json').read_text())
    variants = ARMS[:2]
    forecast_path = folder/'forecast/complete.json'
    forecast = json.loads(forecast_path.read_text())
    inputs = {
        'scripts/v3_2_tilt_size_control.py': digest(ROOT/'scripts/v3_2_tilt_size_control.py'),
        'docs/v3_2_tilt_size_protocol.md': digest(ROOT/'docs/v3_2_tilt_size_protocol.md'),
        'data/v3_2/tilt_predictions/initial/complete.json': digest(BASE/'tilt_predictions/initial/complete.json'),
        'data/v3_2/tilt_predictions/size_control/complete.json': digest(forecast_path),
    }
    source_digests = {
        'data/v3_2/tilt_predictions/size_control/comparison/complete.json': digest(comparison),
        'data/v3_2/tilt_predictions/size_control/comparison/paired.csv': digest(paired),
        'data/v3_2/tilt_predictions/size_control/complete.json': inputs['data/v3_2/tilt_predictions/size_control/complete.json'],
    }
    sample_hashes = set()
    for arm in variants:
        fit = folder/arm
        prefix = f'data/v3_2/tilt_fits/size_control/{arm}'
        fit_record = json.loads((fit/'complete.json').read_text())
        fit_key = f'{prefix}/complete.json'
        fit_digest = digest(fit/'complete.json')
        inputs[fit_key] = source_digests[fit_key] = fit_digest
        sample_sha = fit_record['outputs']['sample.npz']
        sample_hashes.add(sample_sha)
        source_digests[f'{prefix}/sample.npz'] = sample_sha
        if (forecast['inputs'][fit_key] != fit_digest
                or fit_record['all_PSF'] is not True or fit_record['n'] != SIZE
                or (fit_record['case'], fit_record['tilt']) != tuple(arm.split('_'))
                or fit_record['sampler_passed'] is not True or not fit_record['max_rhat'] < 1.05
                or fit_record['minimum_ess'] < 100 or fit_record['divergences'] != 0):
            raise ValueError(f'Invalid size-control fit: {arm}')
        for name in ('entry_sensitivity.csv', 'posterior_entry_sensitivity.csv'):
            source_digests[f'{prefix}/{name}'] = digest(fit/name)
            if source_digests[f'{prefix}/{name}'] != fit_record['outputs'][name]:
                raise ValueError(f'Invalid size-control entry digest: {arm}/{name}')
    if (forecast['all_PSF'] is not True or forecast['n'] != len(objects)
            or forecast['arms'] != list(variants)
            or len(sample_hashes) != 1 or record['inputs'] != inputs
            or record['outputs'] != {'paired.csv': digest(paired)}
            or archived['source_sha256'] != source_digests
            or archived['sample_csv_sha256'] != digest(sample)
            or (record['sample_psf'], record['seed'], record['n']) != (SIZE, SEED, len(objects))
            or record['all_PSF'] is not True or record['new_outcomes_read'] is not False
            or record['grid_flags'] != dict.fromkeys(variants, 0)
            or record['sampler_passed'] != dict.fromkeys(variants, True)):
        raise ValueError('Invalid size-control ancestry or comparison receipt')

    broad = {int(row['row']): row['identity'] for path in sorted((BASE/'tilt_inputs').glob('shard_*/checks.csv'))
             for row in read_csv(path) if row['admission'] == 'broad'}
    sample_rows = read_csv(sample)
    generator = np.random.Generator(np.random.PCG64(SEED))
    expected_rows = np.sort(generator.choice(len(broad), SIZE, replace=False))
    fields = {row['field'] for row in sample_rows}
    if (len(fields) != 13 or [int(row['row']) for row in sample_rows] != expected_rows.tolist()
            or any(row['TYPE'] != 'PSF' or row['identity'] != broad[int(row['row'])]
                   or not isfinite(float(row['design_weight'])) or float(row['design_weight']) <= 0
                   for row in sample_rows)):
        raise ValueError('Archived size-control PSFs differ from the registered broad subsample')

    selected = [group['broad_zero'] for group in objects if group['broad_zero']['candidate_admission'] == 'True']
    expected = {}
    for role in ROLES:
        for stratum, label in (('all', None), ('dwarf', '0'), ('giant', '1')):
            count = sum(row['role'] == role and (label is None or row['class_label'] == label) for row in selected)
            for arm in variants:
                subsample = arm + '_3540'
                for left, right in ((subsample, arm), (arm.replace('broad', 'candidate'), subsample)):
                    expected.update({(role, stratum, left, right, metric): count for metric in METRICS})
    observed = index(read_csv(paired), lambda row: tuple(row[k] for k in ('role', 'stratum', 'left', 'right', 'metric')))
    if (observed.keys() != expected.keys()
            or any(not 0 <= int(row['n']) <= int(row['group_n']) == expected[key]
                   for key, row in observed.items())
            or (archived['sample_psf'], archived['fields'], archived['paired_rows']) != (SIZE, len(fields), len(expected))
            or archived['new_outcomes_read'] is not False):
        raise ValueError('Incomplete size-control paired comparisons')
    return dict(sample_psf=SIZE, fields=len(fields), paired_rows=len(expected),
                comparison_sha256=digest(comparison), sample_sha256=digest(sample),
                archival_audit_sha256=digest(folder/'audit.json'))


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
                size_control=audit_size_control(objects),
                entry_count_bins=len(bins),
                max_abs_log_normalizer=max(abs(r['absolute_log_normalizer_delta']) for r in transfers),
                max_relative_log_weight_span=max(r['relative_log_weight_span'] for r in transfers),
                minimum_weight_ess=min(r['weight_ess'] for r in transfers),
                max_basis_count_delta=max(r['maximum_expected_basis_count_delta'] for r in transfers),
                max_opened_distance_bin_count_delta=max(float(r['maximum_mean_count_delta']) for r in checks.values()),
                scope='Compact output checks. Shared draw covariance stays bound by the allocated report receipt.')


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))

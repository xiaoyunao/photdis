#!/usr/bin/env python3
"""Recompute opened-field paired scores and primary class diagnostics."""
from collections import Counter
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

from photdis.v3_1.validation_report import field_statistic, paired_field_statistic
from v3_2_opened_diagnostics import METRICS, PAIRS, ROLES


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v3_2'
N_BY_ROLE = dict(zip(ROLES, (167, 352)))


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def index(rows, key):
    rows = list(rows)
    result = {key(r): r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate compact evidence row')
    return result


def check_stats(row, expected):
    for name, value in expected.items():
        actual = row[name]
        if value is None:
            passed = actual == ''
        elif isinstance(value, float):
            passed = math.isclose(float(actual), value, abs_tol=1e-9, rel_tol=1e-10)
        else:
            passed = actual == str(value)
        if not passed:
            raise ValueError(f'Recomputed {name} differs from the archived record')


def metric_value(row, metric):
    label = int(row['reference_class'])
    if metric == 'class/correct':
        return float(int(row['predicted_class']) == label) if label >= 0 else math.nan
    key = {'gaia/log_score': 'signed_gaia_log_score', 'class/log_score': 'class_log_score', 'qnear': 'qnear'}[metric]
    value = row[key]
    if not value:
        return -math.inf if key == 'class_log_score' and label >= 0 else math.nan
    return float(value)


def audit():
    sources = {}
    for dataset in ('opened_diagnostics', 'opened_qrf'):
        folder = BASE/dataset
        receipt = folder/'complete.json'
        raw = receipt.read_bytes()
        record = json.loads(raw)
        if record['all_PSF'] is not True:
            raise ValueError(f'Non-PSF source: {dataset}')
        sources[str(receipt.relative_to(ROOT))] = sha256(raw).hexdigest()
        for name, checksum in record['outputs'].items():
            path = folder/name
            if sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f'Invalid source digest: {path}')
            sources[str(path.relative_to(ROOT))] = checksum

    folder = BASE/'opened_diagnostics'
    table = read_csv(folder/'objects.csv')
    groups = index(table, lambda r: (r['role'], int(r['identity']), r['view']))
    identities = {role: [int(r['identity']) for r in table if r['role'] == role and r['view'] == PAIRS[0][0]]
                  for role in ROLES}
    views = {key[-1] for key in groups}
    expected = {(role, i, view) for role, ids in identities.items() for i in ids for view in views}
    if (groups.keys() != expected or {k: len(v) for k, v in identities.items()} != N_BY_ROLE
            or len(views) != 9 or len({i for ids in identities.values() for i in ids}) != 519):
        raise ValueError('Opened PSF identity census differs from the admission')
    for role, ids in identities.items():
        for i in ids:
            primary = groups[role, i, PAIRS[0][0]]
            for view in views:
                row = groups[role, i, view]
                p = [float(row[f'p_{c}']) for c in ('dwarf', 'giant', 'other')]
                if (row['TYPE'] != 'PSF' or row['reference_class'] != primary['reference_class']
                        or row['cell'] != primary['cell'] or min(p) < 0
                        or not math.isclose(sum(p), 1, abs_tol=1e-8)
                        or p[int(row['predicted_class'])] != max(p)):
                    raise ValueError(f'Invalid aligned class predictions: {role}/{view}/{i}')

    paired = index(read_csv(folder/'score_path.csv'),
                   lambda r: tuple(r[k] for k in ('role', 'stratum', 'baseline', 'view', 'metric')))
    if len(paired) != len(ROLES)*3*len(PAIRS)*len(METRICS):
        raise ValueError('Wrong opened paired-score roster')
    census = {}
    for (role, stratum, baseline, view, metric), row in paired.items():
        label = {'all': None, 'dwarf': '0', 'giant': '1'}[stratum]
        ids = [i for i in identities[role] if label is None or groups[role, i, baseline]['reference_class'] == label]
        left = [groups[role, i, view] for i in ids]
        right = [groups[role, i, baseline] for i in ids]
        flags = sum(a['accepted'] != 'True' or b['accepted'] != 'True' for a, b in zip(left, right))
        actual = paired_field_statistic([metric_value(x, metric) for x in left],
                                        [metric_value(x, metric) for x in right],
                                        [int(x['cell']) for x in left])
        check_stats(row, dict(group_n=len(ids), inherited_or_score_flags=flags, **actual))
        if role == ROLES[0] and stratum == 'all':
            previous = census.setdefault(metric, actual['n'])
            if previous != actual['n']:
                raise ValueError('Opened score path changes denominator')
    if (census != {'gaia/log_score': 166, 'class/log_score': 157, 'class/correct': 157, 'qnear': 167}
            or {(b, v) for _, _, b, v, _ in paired} != set(PAIRS)):
        raise ValueError('Wrong paired opened field census or path')

    reliability = read_csv(folder/'reliability.csv')
    confusion_count = 0
    for role, ids in identities.items():
        observed_confusion = index((r for r in read_csv(folder/f'{role}_confusion.csv')
                                    if r['class_source'] == 'class' and r['admission'] == 'all_objects'),
                                   lambda r: (r['view'], r['reference_class'], r['predicted_class']))
        for view in views:
            labels = [groups[role, i, view] for i in ids if int(groups[role, i, view]['reference_class']) >= 0]
            matches = Counter((r['reference_class'], r['predicted_class']) for r in labels)
            totals = Counter(r['reference_class'] for r in labels)
            for (mode, label, predicted), row in observed_confusion.items():
                if mode == view:
                    check_stats(row, dict(n=matches[label, predicted], denominator=totals[label]))
                    confusion_count += 1
            for target, label in (('dwarf', '0'), ('giant', '1')):
                bins = [r for r in reliability if (r['role'], r['view'], r['target_class']) == (role, view, target)]
                if len(bins) != 10 or sum(int(r['n']) for r in bins) != len(labels):
                    raise ValueError(f'Invalid reliability bin census: {role}/{view}/{target}')
                for row in bins:
                    lo, hi = float(row['bin_lower']), float(row['bin_upper'])
                    positions = [(float(r[f'p_{target}']), r) for r in labels]
                    fit = [(p, r) for p, r in positions if lo <= p and (p < hi or p == hi == 1)]
                    prob = sum(p for p, _ in fit)/len(fit) if fit else None
                    actual = field_statistic([float(r['reference_class'] == label) for _, r in fit],
                                             [int(r['cell']) for _, r in fit])
                    check_stats(row, dict(mean_probability=prob, **actual))

    qrf_dir = BASE/'opened_qrf'
    qrf = index(read_csv(qrf_dir/'objects.csv'), lambda r: (r['role'], int(r['identity'])))
    if qrf.keys() != {(role, i) for role, ids in identities.items() for i in ids}:
        raise ValueError('QRF reference cohort differs from opened admission')
    qrf_checks = 0
    for row in read_csv(qrf_dir/'paired.csv'):
        if (row['role'], row['stratum'], row['support'], row['metric']) != (
                ROLES[0], 'all', 'all_QRF_predictions', 'gaia/log_score'):
            continue
        a = [groups[ROLES[0], i, row['view']] for i in identities[ROLES[0]]]
        b = [qrf[ROLES[0], i] for i in identities[ROLES[0]]]
        if not all(y['TYPE'] == 'PSF' and x['cell'] == y['cell'] for x, y in zip(a, b)):
            raise ValueError('QRF pairing changed the sky block or PSF admission')
        actual = paired_field_statistic([metric_value(x, 'gaia/log_score') for x in a],
                                        [float(y['gaia_log_score'] or 'nan') for y in b],
                                        [int(x['cell']) for x in a])
        check_stats(row, actual)
        qrf_checks += 1
    if not qrf_checks:
        raise ValueError('Missing frozen QRF contrasts')
    return dict(passed=True, matched_psf=len(qrf), paired_rows=len(paired), confusion_rows=confusion_count,
                reliability_rows=len(reliability), qrf_signed_gaia_rows=qrf_checks,
                opened_new_sky=census, sources=sources)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))

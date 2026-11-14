#!/usr/bin/env python3
"""Recompute the opened giant-odds mechanism diagnostics from compact rows."""
import json
from math import isclose
from pathlib import Path

from photdis.v3_1.validation_report import paired_field_statistic
from v3_2_factorial_audit import compare_table, verify_folder, digest
from v3_2_opened_audit import check_stats, index, read_csv

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v3_2'
METRICS = ('gaia_score', 'gaia_pit', 'p_giant', 'class_score', 'qnear', 'median_mu',
           'width68', 'width95', 'teacher_score', 'teacher_residual', 'teacher68', 'teacher95')
KEYS = ('role', 'scope', 'label', 'metric', 'baseline')


def value(row, column):
    item = row[column]
    if item in ('True', 'False'):
        return float(item == 'True')
    return float(item or 'nan')


def audit():
    folder = BASE/'branch_development'
    receipt = verify_folder(folder)
    if receipt['new_outcomes_read'] is not False or receipt['class_conditional_invariance'] is not True:
        raise ValueError('Expected the registered opened-only, shape-preserving result')

    source = BASE/'tilt_predictions/initial'
    verify_folder(source)
    frozen = index([r for r in read_csv(source/'objects.csv') if r['view'] == 'candidate_free'],
                   lambda r: int(r['identity']))
    rows = index(read_csv(folder/'objects.csv'), lambda r: int(r['identity']))
    if rows.keys() != frozen.keys() or len(rows) != 526:
        raise ValueError('Wrong opened mechanism PSF census')
    flags = 0
    for identity, row in rows.items():
        original = frozen[identity]
        expected = (row['coarse_scores_passed'] == 'True' and
                    0 <= float(row['coarse_joint_delta']) < .003 and
                    0 <= float(row['coarse_mu_delta']) < .03)
        if (row['TYPE'] != 'PSF' or any(row[k] != original[k] for k in ('role', 'cell', 'candidate_admission'))
                or row['label'] != original['class_label'] or row['numerical_pass'] != str(expected)):
            raise ValueError(f'Wrong PSF identity or numerical status: {identity}')
        for key in ('p_giant', 'qnear'):
            if (not isclose(float(row[f'base_{key}']), float(original[key]), rel_tol=0, abs_tol=1e-10)
                    or not 0 <= float(row[f'adjusted_{key}']) <= 1):
                raise ValueError(f'Wrong mechanism probability: {identity}/{key}')
        flags += not expected
    if flags != receipt['numerical_flags']:
        raise ValueError('Wrong mechanism flag count')

    pairs, confusion = [], []
    counts = index(read_csv(folder/'joint_counts.csv'),
                   lambda r: (r['role'], r['scope'], int(r['label']), int(r['distance_bin'])))
    roles = ('new_sky_field', 'new_source_seen_sky_field')
    scopes = ('all_opened', 'candidate_admitted')
    expected = {(role, scope, label, b) for role in roles for scope in scopes
                for label in range(3) for b in range(6)}
    if counts.keys() != expected:
        raise ValueError('Missing class-distance count bin')
    for role in roles:
        for scope in scopes:
            cohort = [r for r in rows.values() if r['role'] == role and
                      (scope == 'all_opened' or r['candidate_admission'] == 'True')]
            for view in ('base', 'adjusted'):
                masses = [float(r[view]) for k, r in counts.items() if k[:2] == (role, scope)]
                if min(masses) < 0 or not isclose(sum(masses), len(cohort), abs_tol=1e-8, rel_tol=0):
                    raise ValueError(f'Joint count does not conserve cohort: {role}/{scope}/{view}')
                giant = sum(float(r[view]) for k, r in counts.items() if k[:3] == (role, scope, 1))
                if not isclose(giant, sum(float(r[f'{view}_p_giant']) for r in cohort), abs_tol=1e-8, rel_tol=0):
                    raise ValueError(f'Giant count departs from object probabilities: {role}/{scope}/{view}')
            for label in (-1, 0, 1):
                use = [r for r in cohort if label == -1 or int(r['label']) == label]
                cells = [int(r['cell']) for r in use]
                for metric in METRICS:
                    pairs.append(dict(role=role, scope=scope, label=label, metric=metric, baseline='candidate_free',
                                      **paired_field_statistic([value(r, f'adjusted_{metric}') for r in use],
                                                               [value(r, f'base_{metric}') for r in use], cells)))
                if label >= 0:
                    for view in ('base', 'adjusted'):
                        for predicted in range(3):
                            confusion.append(dict(role=role, scope=scope, view=view, label=label, predicted=predicted,
                                                  n=len(use), count=sum(int(r[f'{view}_predicted_class']) == predicted for r in use)))

    # Recompute candidate/free contrasts; the other anchors stay bound by the receipt.
    archived = [r for r in read_csv(folder/'paired.csv') if r['baseline'] == 'candidate_free']
    paired_index = index(archived, lambda row: tuple(row[k] for k in KEYS))
    if paired_index.keys() != {tuple(str(row[k]) for k in KEYS) for row in pairs}:
        raise ValueError('Wrong paired adjusted-score roster')
    for row in pairs:
        check_stats(paired_index[tuple(str(row[k]) for k in KEYS)], row)
    n_confusion = compare_table(folder/'confusion.csv', confusion, ('role', 'scope', 'view', 'label', 'predicted'))
    return dict(job=receipt['job'], receipt_sha256=digest(folder/'complete.json'),
                opened_psf=len(rows), flags=flags, candidate_free_pairs=len(pairs),
                confusion_cells=n_confusion, count_bins=len(counts))


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))

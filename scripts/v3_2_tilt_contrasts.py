#!/usr/bin/env python3
"""Direct paired factorial contrasts in the opened v3.1 field."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from photdis.v3_1.validation_report import field_statistic

ARMS = ('broad_zero', 'broad_free', 'candidate_zero', 'candidate_free')
METRICS = ('signed_gaia_log_score', 'class_log_score', 'p_giant', 'qnear',
           'oracle_gaia_score', 'oracle_astronn_score', 'oracle_far_mass',
           'oracle_median_mu', 'oracle_far_peak_mu')
CONTRASTS = {
    'broad_free_minus_zero': (-1, 1, 0, 0),
    'candidate_free_minus_zero': (0, 0, -1, 1),
    'candidate_minus_broad_zero': (-1, 0, 1, 0),
    'candidate_minus_broad_free': (0, -1, 0, 1),
    'population_by_tilt_interaction': (1, -1, -1, 1),
    'candidate_zero_minus_broad_free': (0, -1, 1, 0),
}


def number(row, key):
    if key == 'class_log_score' and int(row['class_label']) >= 0 and not row.get(key):
        return -np.inf
    return float(row[key]) if row.get(key) else np.nan


def matched_rows(source):
    objects = {}
    for row in source:
        if row['TYPE'] != 'PSF':
            raise ValueError('Verified PSF required in all four arms')
        key = int(row['identity'])
        group = objects.setdefault(key, {})
        if row['view'] in group:
            raise ValueError(f'Duplicate view for {key}')
        group[row['view']] = row
    for key, group in objects.items():
        if set(group) != set(ARMS):
            raise ValueError(f'Incomplete factorial for {key}')
        for field in ('role', 'cell', 'class_label', 'candidate_admission'):
            if len({row[field] for row in group.values()}) != 1:
                raise ValueError(f'Mismatched {field} for {key}')
    return [objects[key] for key in sorted(objects)]


def contrast_rows(groups):
    rows = []
    for role in ('new_sky_field', 'new_source_seen_sky_field'):
        selected = [g for g in groups if g[ARMS[0]]['role'] == role
                    and g[ARMS[0]]['candidate_admission'] == 'True']
        for stratum, label in [('all', None), ('dwarf', 0), ('giant', 1)]:
            group = [g for g in selected if label is None or int(g[ARMS[0]]['class_label']) == label]
            cells = np.array([int(g[ARMS[0]]['cell']) for g in group])
            for metric in METRICS:
                values = np.array([[number(g[a], metric) for a in ARMS] for g in group], dtype=float)
                values = values.reshape(-1, len(ARMS))
                for name, coefficient in CONTRASTS.items():
                    c = np.asarray(coefficient)
                    use = c != 0
                    pair = values[:, use]
                    available = ~np.isnan(pair).any(axis=1)
                    with np.errstate(invalid='ignore'):
                        delta = pair[available] @ c[use]
                    undefined = np.isnan(delta)
                    delta[undefined] = np.inf
                    rows.append(dict(role=role, stratum=stratum, metric=metric, contrast=name,
                                     group_n=len(group), undefined_nonfinite=int(undefined.sum()),
                                     **field_statistic(delta, cells[available])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('forecast', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    path = args.forecast / 'objects.csv'
    marker = json.loads((args.forecast / 'complete.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != marker['outputs'][path.name]:
        raise ValueError('Forecast source hash mismatch')
    with path.open(newline='') as stream:
        groups = matched_rows(csv.DictReader(stream))
    if len(groups) != marker['n']:
        raise ValueError('Forecast identity count mismatch')
    rows = contrast_rows(groups)
    with args.destination.open('x', newline='') as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()

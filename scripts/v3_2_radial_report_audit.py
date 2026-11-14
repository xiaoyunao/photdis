#!/usr/bin/env python3
"""Reconcile the compact radial recovery report with the frozen mock design."""
import json
from math import isclose
from pathlib import Path

from v3_2_radial_cache_audit import audit as cache_audit
from v3_2_radial_report import (count_loss_comparison, empirical_penalty, grouped_summary,
                                read_csv, shrinkage_summary, validate_interval, validate_numerical,
                                validate_truth)
from v2_1_near import ROOT, sha


BASE = ROOT/'manifests/v3_2'
GATES = ('broad', 'local')
LOCAL = ('mu_location', 'mu_width', 'pre_entry_fraction', 'selected_members')


def matching_table(path, rows):
    archived = read_csv(path)
    if not rows and archived == [{'status': 'no rows'}]:
        return
    if len(archived) != len(rows):
        raise ValueError(f'Changed table length: {path.name}')
    if not rows:
        return
    keys = {key for row in rows for key in row}
    for observed, expected in zip(archived, rows):
        if set(observed) != keys:
            raise ValueError(f'Changed table fields: {path.name}')
        for key, raw in observed.items():
            value = expected.get(key)
            if value is None:
                valid = raw == ''
            elif isinstance(value, float):
                valid = isclose(float(raw), value, rel_tol=1e-10, abs_tol=1e-12)
            else:
                valid = raw == str(value)
            if not valid:
                raise ValueError(f'Changed {key} in {path.name}: {raw} vs {value}')


def audit_rows(folder, design, receipt):
    gates = read_csv(folder/'numerical.csv')
    for row in gates:
        row['catalog'] = int(row['catalog'])
    indexed = {(r['catalog'], r['arm']): r for r in gates}
    ids = {int(row['catalog']) for row in design}
    if len(ids) != len(design) or len(gates) != len(indexed) or indexed.keys() != {(i, a) for i in ids for a in GATES}:
        raise ValueError('Missing or duplicate radial verdict')
    accepted = {}
    for key, row in indexed.items():
        accepted[key] = validate_numerical(row)
        if row['accepted'] != str(accepted[key]):
            raise ValueError(f'Changed numerical verdict: {key}')
    recovery = [r for r in read_csv(folder/'recovery.csv') if r.get('arm')]
    recovered = {}
    for row in recovery:
        row['catalog'] = int(row['catalog'])
        key = row['catalog'], row['arm']
        if row['accepted'] != str(accepted[key]):
            raise ValueError(f'Recovery and verdict disagree: {key}')
        row['accepted'] = accepted[key]
        validate_interval(row)
        metrics = recovered.setdefault(key, {})
        if row['metric'] in metrics:
            raise ValueError(f'Duplicate recovery metric: {key}')
        metrics[row['metric']] = row
    for row in design:
        i = int(row['catalog'])
        for arm in GATES:
            expected = set()
            if indexed[i, arm]['status'] == 'sampled':
                expected = {f'count_{c}_{b}' for c in range(3) for b in range(6)}
                if arm == 'local':
                    expected.update(LOCAL)
            if recovered.get((i, arm), {}).keys() != expected:
                raise ValueError(f'Incomplete metrics for catalog {i}/{arm}')
        validate_truth([r for arm in GATES for r in recovered.get((i, arm), {}).values()], row)

    paired, paired_summary = count_loss_comparison(design, recovery)
    shrinkage, shrinkage_grouped = shrinkage_summary(design, recovery)
    empirical, empirical_summary = empirical_penalty(design, recovery, paired)
    tables = {
        'coverage': grouped_summary(design, recovery), 'paired_count_loss': paired,
        'paired_count_summary': paired_summary, 'shrinkage': shrinkage,
        'shrinkage_summary': shrinkage_grouped, 'empirical_minus_model': empirical,
        'empirical_minus_model_summary': empirical_summary,
    }
    for name, rows in tables.items():
        matching_table(folder/f'{name}.csv', rows)
    terms = dict(n_catalogs=len(design), n_arms=len(gates), relative_and_counts_accepted=sum(accepted.values()),
                 both_fitted=len(paired), both_accepted=sum(r['both_accepted'] for r in paired))
    if any(receipt[key] != value for key, value in terms.items()):
        raise ValueError('Aggregate census differs from the radial report receipt')
    return dict(**terms, recomputed_rows={name: len(rows) for name, rows in tables.items()})


def audit():
    cache_audit(complete=True)
    folder = BASE/'radial_report'
    path = folder/'complete.json'
    receipt = json.loads(path.read_text())
    if receipt['all_PSF'] is not True:
        raise ValueError('Expected the verified PSF recovery report')
    artifacts = {'numerical', 'recovery', 'coverage', 'paired_count_loss', 'paired_count_summary',
                 'shrinkage', 'shrinkage_summary', 'empirical_minus_model', 'empirical_minus_model_summary'}
    if receipt['outputs'].keys() != {f'{name}.csv' for name in artifacts}:
        raise ValueError('Unexpected compact radial report inventory')
    for name, digest in receipt['outputs'].items():
        if sha(folder/name) != digest:
            raise ValueError(f'Invalid radial report digest: {name}')
    design_folder = BASE/'radial_design'
    design_path = design_folder/'complete.json'
    design_outputs = json.loads(design_path.read_text())['outputs']
    design_digest = sha(design_path)
    design = read_csv(design_folder/'design.csv')
    if sorted(int(r['catalog']) for r in design) != list(range(288)):
        raise ValueError('Incomplete frozen radial design')
    expected_sources = {f'data/v3_2/radial_recovery/catalog_{i:03}/complete.json' for i in range(288)}
    expected_sources.update({
        'data/v3_2/radial_design/complete.json', 'data/v3_2/radial_design/design.csv',
        'scripts/v3_2_radial_report.py', 'src/photdis/v3_1/comparison.py',
        'src/photdis/v3_1/radial.py',
    })
    if receipt['inputs'].keys() != expected_sources:
        raise ValueError('Incomplete or unexpected radial report ancestry')
    for name, digest in receipt['inputs'].items():
        source = ROOT/name.replace('data/v3_2/', 'manifests/v3_2/', 1)
        if sha(source) != digest:
            raise ValueError(f'Invalid radial report ancestor: {name}')
    by_catalog = {name: {} for name in ('numerical.csv', 'recovery.csv')}
    for name, catalogs in by_catalog.items():
        for row in read_csv(folder/name):
            if row.get('arm'):
                catalogs.setdefault(int(row['catalog']), []).append(row)
    for row in design:
        i, repeat = int(row['catalog']), int(row['repeat'])
        source = BASE/f'radial_recovery/catalog_{i:03}/complete.json'
        result = json.loads(source.read_text())
        if result['catalog'] != i or result['all_PSF'] is not True or result['n'] != int(row['n']):
            raise ValueError(f'Recovery ancestry does not declare the registered census: {i}')
        if result['inputs']['data/v3_2/radial_design/complete.json'] != design_digest:
            raise ValueError(f'Recovery uses a different design receipt: {i}')
        for kind in ('photometry', 'truth'):
            name = f'{kind}_{i:03}.npz'
            if result['inputs'][f'data/v3_2/radial_design/{name}'] != design_outputs[name]:
                raise ValueError(f'Recovery uses a different {kind} payload: {i}')
        if i != 0:
            cache = f'radial_entry/repeat_{repeat:02}/complete.json'
            if result['inputs'][f'data/v3_2/{cache}'] != sha(BASE/cache):
                raise ValueError(f'Recovery uses a different cache: {i}')
        for name, catalogs in by_catalog.items():
            part = source.with_name(name)
            if sha(part) != result['outputs'][name]:
                raise ValueError(f'Invalid catalog result digest: {i}/{name}')
            original = [r for r in read_csv(part) if r.get('arm')]
            pooled = catalogs.get(i, [])
            if len(original) != len(pooled) or any(
                any(value != b[key] for key, value in a.items()) for a, b in zip(original, pooled)
            ):
                raise ValueError(f'Aggregate differs from catalog result: {i}/{name}')
            if name == 'numerical.csv' and result['fitted_arms'] != sorted(
                r['arm'] for r in original if r['status'] == 'sampled'
            ):
                raise ValueError(f'Fitted arms differ from catalog receipt: {i}')
    return dict(job=receipt['job'], receipt_sha256=sha(path), **audit_rows(folder, design, receipt))


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))

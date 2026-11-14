#!/usr/bin/env python3
"""Aggregate frozen narrow-structure coverage with retained failure denominators."""
import csv
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from photdis.stage1 import require_compute
from photdis.v3_1.comparison import paired_mean
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_2_tilt_inputs import verified


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def coverage(rows, total):
    errors = np.array([float(r['median']) - float(r['truth']) for r in rows])
    k = sum(r['covered95'] == 'True' for r in rows)
    n = len(rows)
    stat = dict(covered=k, fitted=n, registered=total,
                coverage_lower_all=k/total, coverage_upper_all=(k+total-n)/total)
    if n:
        interval = binomtest(k, n).proportion_ci(method='exact')
        stat.update(coverage=k/n, coverage_lo95=float(interval.low),
                    coverage_hi95=float(interval.high), mean_bias=float(errors.mean()),
                    mean_absolute_error=float(abs(errors).mean()),
                    mean_interval_length=float(np.mean([float(r['interval_length']) for r in rows])))
    return stat


def design_groups(designs):
    groups = defaultdict(list)
    for d in designs:
        key = (d['design'], d['source'], d['background'])
        if d['design'] == 'fixed':
            key += (d['fraction'], d['width'])
        else:
            key += ('', '')
        groups[key].append(int(d['catalog']))
    return groups


def grouped_summary(designs, rows):
    by_catalog = defaultdict(list)
    for r in rows:
        by_catalog[int(r['catalog'])].append(r)
    metrics = [(arm, f'count_{c}_{b}') for arm in ('broad', 'local')
               for c in range(3) for b in range(6)]
    metrics += [('local', k) for k in ('mu_location', 'mu_width', 'pre_entry_fraction', 'selected_members')]
    results = []
    for (design, source, background, fraction, width), catalogs in design_groups(designs).items():
        data = [r for i in catalogs for r in by_catalog[i]]
        for arm, key in metrics:
            observations = [r for r in data if (r['arm'], r['metric']) == (arm, key)]
            for view, use in [('all_fitted', observations),
                              ('relative_and_counts_accepted', [r for r in observations if r['accepted']])]:
                results.append(dict(design=design, source=source, background=background,
                                    fraction=fraction, width=width, arm=arm, metric=key, subset=view,
                                    **coverage(use, len(catalogs))))
    return results


def count_loss_comparison(designs, rows):
    """One paired loss contrast per catalog, preserving all 18 correlated bins."""
    count_keys = {f'count_{c}_{b}' for c in range(3) for b in range(6)}
    by_arm = defaultdict(dict)
    for row in rows:
        if row['metric'] in count_keys:
            arm = by_arm[row['catalog'], row['arm']]
            if row['metric'] in arm:
                raise ValueError('Duplicate count bin')
            arm[row['metric']] = row
    metadata = {int(d['catalog']): d for d in designs}
    object_rows = {}
    for catalog, design in metadata.items():
        broad, local = [by_arm[catalog, arm] for arm in ('broad', 'local')]
        if not broad or not local:
            continue
        if set(broad) != count_keys or set(local) != count_keys:
            raise ValueError(f'Missing joint count bins for catalog {catalog}')
        keys = sorted(count_keys)
        truth = np.array([float(broad[k]['truth']) for k in keys])
        np.testing.assert_array_equal(truth, [float(local[k]['truth']) for k in keys])
        n = int(design['n'])
        if np.any(truth < 0) or not np.all(truth == np.floor(truth)) or truth.sum() != n:
            raise ValueError(f'Invalid joint count truth for catalog {catalog}')
        errors = [float(abs(np.array([float(arm[k]['median']) for k in keys])-truth).sum()/n)
                  for arm in (broad, local)]
        widths = [float(np.mean([float(arm[k]['interval_length']) for k in keys]))
                  for arm in (broad, local)]
        accepted = all(r['accepted'] for arm in (broad, local) for r in arm.values())
        object_rows[catalog] = dict(catalog=catalog, repeat=int(design['repeat']), both_accepted=accepted,
                                    broad_count_l1=errors[0], local_count_l1=errors[1],
                                    count_l1_change=errors[1]-errors[0],
                                    broad_mean_interval=widths[0], local_mean_interval=widths[1],
                                    mean_interval_change=widths[1]-widths[0])
    summary = []
    for (design, source, background, fraction, width), catalogs in design_groups(designs).items():
        fitted = [object_rows[i] for i in catalogs if i in object_rows]
        for subset, selected in [('both_fitted', fitted),
                                 ('both_accepted', [r for r in fitted if r['both_accepted']])]:
            for metric in ('count_l1_change', 'mean_interval_change'):
                stats = paired_mean([r[metric] for r in selected], np.ones(len(selected)),
                                    [r['repeat'] for r in selected], seed=320511)
                stats['repeats'] = stats.pop('cells')
                summary.append(dict(design=design, source=source, background=background, fraction=fraction,
                                    width=width, subset=subset, metric=metric, registered=len(catalogs),
                                    **stats))
    return list(object_rows.values()), summary


def main():
    require_compute()
    source = ROOT / 'data/v3_2/radial_design'
    marker = source / 'complete.json'
    design = json.loads(marker.read_text())
    table = verified(source / 'design.csv', design['outputs']['design.csv'])
    designs = read_csv(table)
    if len(designs) != 288 or sorted(int(d['catalog']) for d in designs) != list(range(288)):
        raise ValueError('Incomplete frozen design')
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in
              (marker, table, Path(__file__).resolve(), ROOT/'src/photdis/v3_1/comparison.py')}
    gates, recovery = [], []
    for d in designs:
        catalog = int(d['catalog'])
        folder = ROOT / 'data/v3_2/radial_recovery' / f'catalog_{catalog:03}'
        path = folder / 'complete.json'
        result = json.loads(path.read_text())
        if not result['all_PSF'] or result['catalog'] != catalog or result['n'] != int(d['n']):
            raise ValueError(f'Unexpected catalog identity: {catalog}')
        if result['inputs'][str(marker.relative_to(ROOT))] != inputs[str(marker.relative_to(ROOT))]:
            raise ValueError(f'Design mismatch: {catalog}')
        inputs[str(path.relative_to(ROOT))] = sha(path)
        status = read_csv(verified(folder / 'numerical.csv', result['outputs']['numerical.csv']))
        accepted = {}
        for r in status:
            # Keep the absolute objective drift separate from the conditional
            # relative shift, count transfer, and sampler gates.
            accepted[r['arm']] = all(r.get(k) == 'True' for k in (
                'sampler_pass', 'entry_pass', 'relative_objective_pass', 'fixed_draw_count_pass'))
            gates.append(dict(catalog=catalog, accepted=accepted[r['arm']], **r))
        if set(accepted) != {'local', 'broad'} or len(status) != 2:
            raise ValueError(f'Missing arm status: {catalog}')
        rows = read_csv(verified(folder / 'recovery.csv', result['outputs']['recovery.csv']))
        seen = set()
        for r in rows:
            if 'arm' not in r:
                continue
            k = r['arm'], r['metric']
            if k in seen:
                raise ValueError(f'Duplicate recovery metric: {catalog} {k}')
            seen.add(k)
            recovery.append(dict(catalog=catalog, accepted=accepted[r['arm']], **r))
        fitted = {r['arm'] for r in status if r['status'] == 'sampled'}
        expected = {(arm, f'count_{c}_{b}') for arm in fitted for c in range(3) for b in range(6)}
        if 'local' in fitted:
            expected |= {('local', name) for name in ('mu_location', 'mu_width',
                                                      'pre_entry_fraction', 'selected_members')}
        if seen != expected or fitted != set(result['fitted_arms']):
            raise ValueError(f'Incomplete fitted recovery records: {catalog}')
    summary = grouped_summary(designs, recovery)
    paired, paired_summary = count_loss_comparison(designs, recovery)
    output = ROOT / 'data/v3_2/radial_report'
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('numerical', gates), ('recovery', recovery), ('coverage', summary),
                       ('paired_count_loss', paired), ('paired_count_summary', paired_summary)]:
        csv_rows(output / f'{name}.csv', rows)
    save_json(output / 'complete.json', dict(
        inputs=inputs, outputs={p.name: sha(p) for p in sorted(output.iterdir())},
        all_PSF=True, n_catalogs=len(designs), n_arms=len(gates),
        relative_and_counts_accepted=sum(r['accepted'] for r in gates),
        both_fitted=len(paired), both_accepted=sum(r['both_accepted'] for r in paired),
        job=os.getenv('SLURM_JOB_ID'),
        scope='Binomial Monte Carlo intervals conditional on fixed contexts/source library. '
              'Fixed-truth recovery and prior-predictive coverage are separate.'))


if __name__ == '__main__':
    main()

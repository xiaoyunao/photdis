#!/usr/bin/env python3
"""Reconcile collected sky-transfer records while retaining failed numerics."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from v2_1_near import ROOT, sha, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-states', type=int, default=102,
                        help='Explicit partial inventory size; final delivery requires 102')
    expected = parser.parse_args().expected_states
    assert 0 < expected <= 102
    base = ROOT / 'manifests/v3_1/heldout_transfer'
    inputs, checks, inventory = {}, [], []

    def check(name, value):
        checks.append(dict(check=name, passed=bool(value)))
        if not value:
            raise AssertionError(name)

    for case in ['broad', 'candidate', 'holdout']:
        for path in sorted((base / case).glob('correction_*/complete.json')):
            done = json.loads(path.read_text())
            index = int(path.parent.name.split('_')[1])
            key = f'{case}/{index}'
            check('identity:' + key, done['case'] == case and done['correction'] == index and
                  0 <= index < 34 and done['available'])
            check('PSF_and_sample:' + key, done['all_PSF'] and done['heldout_rows'] == 6000 and
                  done['candidate_rows'] == 935 and not done['references_read'])
            fields, fitted = set(done['heldout_fields']), set(done['fitted_fields'])
            check('sky_role:' + key, fields == {4, 6, 12} and
                  done['population_fit_excludes_heldout'] == (case == 'holdout') and
                  (fields.isdisjoint(fitted) if case == 'holdout' else fields <= fitted))
            inputs[str(path.relative_to(ROOT))] = sha(path)
            for name in ['scripts/v3_1_heldout_transfer.py', 'src/photdis/v3_1/predictive.py',
                         'docs/v3_1_heldout_transfer_protocol.md']:
                check('frozen_source:' + key + ':' + name, sha(ROOT / name) == done['inputs'][name])
                inputs[name] = sha(ROOT / name)
            tables = {}
            for name in ['checks.csv', 'predictive_summary.csv']:
                p = path.parent / name
                check('output:' + key + ':' + name, sha(p) == done['outputs'][name])
                inputs[str(p.relative_to(ROOT))] = sha(p)
                with p.open() as stream:
                    tables[name] = list(csv.DictReader(stream))
            variants = {(e, v) for e in ['broad', 'candidate']
                        for v in ['base', 'entry_seed', 'entry_level', 'native_fine']}
            for name, rows in tables.items():
                check('eight_variants:' + key + ':' + name, len(rows) == 8 and
                      {(r['envelope'], r['variant']) for r in rows} == variants)
            rows = tables['checks.csv']
            for row in rows:
                passed = (float(row['maximum_probability_delta']) < .003 and
                          float(row['maximum_pointwise_logscore_delta']) < .02 and
                          float(row['maximum_count_mean_delta']) < .5 and
                          float(row['marginal_half_delta']) < .005)
                check('original_gate:' + key + ':' + row['envelope'] + ':' + row['variant'],
                      passed == (row['passed'] == 'True'))
            check('original_outcome:' + key, done['passed'] == all(r['passed'] == 'True' for r in rows))
            check('predictive_denominators:' + key, all(int(r['n']) ==
                  {'broad': 6000, 'candidate': 935}[r['envelope']] for r in tables['predictive_summary.csv']))
            inventory.append(dict(case=case, correction=index, passed=done['passed'],
                numeric_failures=sum(r['passed'] != 'True' for r in rows),
                marker=str(path.relative_to(ROOT)), sha256=sha(path)))
    states = {(r['case'], r['correction']) for r in inventory}
    check('expected_unique_states', len(inventory) == len(states) == expected)
    remaining = [dict(case=case, correction=index) for case in ['broad', 'candidate', 'holdout']
                 for index in range(34) if (case, index) not in states]
    if expected == 102:
        check('complete_registered_inventory', not remaining)
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    save_json(base / f'compact_inventory_{expected}_states.json', dict(inputs=inputs,
        actual_checks=len(checks), checks=checks, compact_consistency_passed=True,
        completed_states=expected, all102_states_present=not remaining,
        remaining_states=remaining, inventory=inventory,
        scope='Collected compact outputs, exact source/sky roles and original numerical gates. Failed states remain. No fresh large-payload audit, science acceptance, independent distance validation or final figure acceptance.'))
    print(json.dumps(dict(states=expected, actual_checks=len(checks),
        by_case=dict(Counter(r['case'] for r in inventory)),
        failed_states=sum(not r['passed'] for r in inventory),
        failed_checks=sum(r['numeric_failures'] for r in inventory))))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Reconcile original/refined entry flags and execution provenance, locally."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-states', type=int, default=32)
    expected = parser.parse_args().expected_states
    assert 1 <= expected <= 32
    root = ROOT / 'manifests/v3_1/structure'
    markers = sorted((root / 'entry_refined').glob('draw_??/complete.json'))
    if len(markers) != expected:
        raise ValueError(f'Expected {expected} states; found {len(markers)}. No output written.')
    inputs, checks, summary = {}, [], []
    common_ids = None

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(dict(check=name, passed=True))

    def record(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)

    for marker in markers:
        draw = int(marker.parent.name.split('_')[1])
        prefix = f'draw_{draw:02}:'
        done = json.loads(marker.read_text())
        original = root / f'entry/draw_{draw:02}/complete.json'
        old = json.loads(original.read_text())
        table, old_table = marker.parent / 'summary.csv', original.parent / 'checks.csv'
        with table.open() as stream:
            rows = list(csv.DictReader(stream))
        with old_table.open() as stream:
            old_rows = list(csv.DictReader(stream))
        for path in [marker, original, table, old_table]:
            record(path)
        check(prefix + 'state', done['response_draw'] == old['response_draw'] == draw and 0 <= draw < 32)
        check(prefix + 'table_hashes', sha(table) == done['outputs']['summary.csv'] and
              sha(old_table) == old['outputs']['checks.csv'])
        original_key = f'data/v3_1/structure/entry/draw_{draw:02}/complete.json'
        check(prefix + 'original_marker', done['inputs'][original_key] == sha(original))
        shared = [f'data/v3_0/uncertainty_refit/draw_{draw:02}/model.joblib',
                  'data/v3_0/uncertainty/shared_draws.npz', 'data/v3_1/structure/photometry.npz']
        check(prefix + 'unchanged_data_and_model', all(done['inputs'][n] == old['inputs'][n] for n in shared))
        for name, digest in done['inputs'].items():
            if name.startswith(('scripts/', 'src/', 'docs/')):
                check(prefix + name, sha(ROOT / name) == digest)
                record(ROOT / name)
        check(prefix + 'native_PSF', done['n'] == old['n'] == 512 and done['all_PSF'] and old['all_PSF'] and
              all(r['TYPE'] == 'PSF' for r in rows + old_rows))
        expected_pairs = {(i, b) for i in range(512) for b in ['background', 'structure']}
        pairs = {(int(r['row']), r['branch']) for r in rows}
        check(prefix + 'all_branches', len(rows) == len(old_rows) == 1024 and pairs == expected_pairs and
              {(int(r['row']), r['branch']) for r in old_rows} == expected_pairs)
        ids = {(int(r['row']), int(r['identity'])) for r in rows}
        if common_ids is None:
            common_ids = ids
        check(prefix + 'same512identities', ids == common_ids ==
              {(int(r['row']), int(r['identity'])) for r in old_rows} and
              len(ids) == len({identity for _, identity in ids}) == 512)
        check(prefix + 'complete_contexts', done['rows'] == list(range(512)) and
              all(int(r['context']) == int(r['row']) for r in rows) and
              set(done['contexts']) == {f'context_{i}/complete.json' for i in range(512)})
        original_flags = {(int(r['row']), r['branch']): r['passed'] for r in old_rows}
        check(prefix + 'original_flags_preserved', all(r['original_passed'] ==
              original_flags[int(r['row']), r['branch']] for r in rows))
        failed_original, unresolved = set(), set()
        gates, unchanged = [], []
        for row in rows:
            check_values = [row.get(k, '') for k in ['draws', 'previous_delta', 'previous_relative',
                            'independent_delta', 'independent_relative', 'physical', 'minimum', 'maximum']]
            if row['original_passed'] == 'True':
                unchanged.append(row['passed'] == 'True' and not any(check_values))
            else:
                failed_original.add(int(row['row']))
                values = {k: float(row[k]) for k in ['previous_delta', 'previous_relative',
                          'independent_delta', 'independent_relative', 'minimum', 'maximum']}
                physical = values['maximum'] <= 1
                passed = (values['previous_delta'] < .0005 and values['independent_delta'] < .0005 and
                          values['previous_relative'] < .01 and values['independent_relative'] < .01 and physical)
                gates.append(int(row['draws']) == 8388608 and all(math.isfinite(v) for v in values.values()) and
                             row['physical'] == str(physical) and row['passed'] == str(passed))
            if row['passed'] != 'True':
                unresolved.add(int(row['row']))
        check(prefix + 'passing_branches_unchanged', all(unchanged))
        check(prefix + 'bounded_original_gates', all(gates))
        check(prefix + 'original_failure_count', len(failed_original) == old['failures'] == done['original_failures'])
        check(prefix + 'refined_outcome', len(unresolved) == done['unresolved'] and done['passed'] == (not unresolved))
        workers, elapsed = 8, None
        if 'resource_execution' in done:
            audit = ROOT / done['resource_execution'].replace('data/v3_1/', 'manifests/v3_1/', 1)
            execution = json.loads(audit.read_text())
            record(audit)
            check(prefix + 'execution_hash', sha(audit) == done['resource_execution_sha256'])
            check(prefix + 'execution_identity', execution['scientific_inputs'] == done['inputs'] and
                  execution['scientific_source_sha256'] == done['inputs']['scripts/v3_1_structure_entry_refine.py'] and
                  execution['job'] == done['job'] and execution['execution_complete'])
            workers = execution['workers']
            source = 'scripts/v3_1_structure_entry_resume' + (str(workers) if workers in [48, 64] else '') + '.py'
            check(prefix + 'execution_source', workers in [32, 48, 64] and
                  sha(ROOT / source) == execution['execution_source_sha256'])
            record(ROOT / source)
            check(prefix + 'memory_guard', workers * execution['planning_bound_per_worker_GiB'] * 1024**3 <=
                  .7 * execution['memory_limit_bytes'])
            elapsed = execution['elapsed_seconds']
        else:
            check(prefix + 'original_eight_worker_execution', draw == 1 and done['job'] == '61223174')
        summary.append(dict(draw=draw, n=512, original_failed_contexts=len(failed_original),
                            unresolved_contexts=len(unresolved), refined_passed=done['passed'],
                            workers=workers, execution_seconds=elapsed, job=done['job']))
    check('unique_state_inventory', len({r['draw'] for r in summary}) == expected and
          (expected != 32 or {r['draw'] for r in summary} == set(range(32))))
    record(Path(__file__))
    name = f'refined_entry_{expected}_states'
    table = root / (name + '.csv')
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    result = dict(inputs=inputs, outputs={table.name: sha(table)}, states=expected,
                  all32_present=expected == 32, checked_branch_rows=1024 * expected,
                  original_failed_contexts=sum(r['original_failed_contexts'] for r in summary),
                  unresolved_contexts=sum(r['unresolved_contexts'] for r in summary),
                  actual_checks=len(checks), checks=checks, compact_consistency_passed=True,
                  scientific_acceptance=False,
                  scope='Compact original/refined tables and execution provenance only. Original failures retained; partial inventory is explicit. Full payload authentication remains in allocated assembly; no posterior or scientific acceptance.')
    (root / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['states', 'actual_checks', 'original_failed_contexts', 'unresolved_contexts']}))


if __name__ == '__main__':
    main()

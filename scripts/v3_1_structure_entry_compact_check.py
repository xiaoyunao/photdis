#!/usr/bin/env python3
"""Audit all 32 original entry summaries without replacing their failures."""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = ROOT / 'manifests/v3_1/structure'
    markers = [root / f'entry/draw_{i:02}/complete.json' for i in range(32)]
    missing = [str(p.relative_to(ROOT)) for p in markers if not p.exists()]
    if missing:
        raise FileNotFoundError('All 32 original entry states required: ' + ', '.join(missing))
    inputs, checks, summary = {}, [], []
    common_ids = common_photometry = None

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(dict(check=name, passed=True))

    for draw, marker in enumerate(markers):
        done = json.loads(marker.read_text())
        table = marker.parent / 'checks.csv'
        rows = list(csv.DictReader(table.open()))
        for path in [marker, table]:
            inputs[str(path.relative_to(ROOT))] = sha(path)
        prefix = f'draw_{draw:02}:'
        check(prefix + 'table_hash', sha(table) == done['outputs']['checks.csv'])
        check(prefix + 'state_identity', done['response_draw'] == done['inputs']['response_draw'] == draw)
        check(prefix + '512PSF', done['n'] == 512 and done['all_PSF'] and
              all(r['TYPE'] == 'PSF' for r in rows))
        pairs = {(int(r['row']), r['branch']) for r in rows}
        check(prefix + '1024_branches', len(rows) == 1024 and pairs ==
              {(i, b) for i in range(512) for b in ['background', 'structure']})
        ids = {(int(r['row']), int(r['identity'])) for r in rows}
        check(prefix + 'unique_identities', len(ids) == len({i for _, i in ids}) == 512)
        photometry = done['inputs']['data/v3_1/structure/photometry.npz']
        if common_ids is None:
            common_ids, common_photometry = ids, photometry
        check(prefix + 'same_objects_and_photometry', ids == common_ids and photometry == common_photometry)
        check(prefix + 'all_context_markers', set(done['contexts']) ==
              {f'context_{i}/complete.json' for i in range(512)})
        for name in ['scripts/v3_1_structure_entry.py', 'scripts/v3_1_structure_entry_pilot.py',
                     'src/photdis/v3_1/structure_entry.py']:
            check(prefix + name, sha(ROOT / name) == done['inputs'][name])
            inputs[name] = sha(ROOT / name)
        physical_matches, gate_matches = [], []
        for r in rows:
            physical = float(r['maximum']) <= 1
            passed = bool(r['previous_delta']) and physical and (
                float(r['previous_delta']) < .0005 and float(r['independent_delta']) < .0005 and
                float(r['previous_relative']) < .01 and float(r['independent_relative']) < .01)
            physical_matches.append(physical == (r['physical'] == 'True'))
            gate_matches.append(passed == (r['passed'] == 'True'))
        check(prefix + 'all1024_physical_flags', all(physical_matches))
        check(prefix + 'all1024_original_gates', all(gate_matches))
        failed = {int(r['row']) for r in rows if r['passed'] != 'True'}
        check(prefix + 'original_failure_count', len(failed) == done['failures'])
        check(prefix + 'original_outcome', done['passed'] == (not failed))
        summary.append(dict(draw=draw, n=512, original_failed_contexts=len(failed),
                            background_failed=sum(r['branch'] == 'background' and r['passed'] != 'True' for r in rows),
                            structure_failed=sum(r['branch'] == 'structure' and r['passed'] != 'True' for r in rows),
                            original_passed=done['passed'], job=done['job']))
    path = root / 'initial_entry_summary.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    record = dict(inputs=inputs, outputs={path.name: sha(path)}, states=32, contexts=16384,
                  original_failed_contexts=sum(r['original_failed_contexts'] for r in summary),
                  actual_checks=len(checks), checked_branch_rows=32768,
                  checks=checks, compact_consistency_passed=True,
                  scientific_acceptance=False,
                  scope='Original compact summaries only: sources, same512PSF identities, all32states and original numerical flags. Full context payload authentication remains in allocated assembly; no refinement, posterior or science acceptance.')
    (root / 'initial_entry_inventory.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ['states', 'contexts', 'original_failed_contexts', 'actual_checks']}))


if __name__ == '__main__':
    main()

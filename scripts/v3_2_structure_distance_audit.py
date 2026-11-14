#!/usr/bin/env python3
"""Verify compact NGC288 distance-node receipts before joint interpretation."""
import argparse
import csv
from hashlib import sha256
import io
import json
from math import isfinite
from pathlib import Path

from v3_2_collect_radial_results import archive_verified, collect, discover


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v3_2/structure_distance'
COMPACT_FILES = ('complete.json', 'checks.csv', 'inputs.json')
TOLERANCES = {
    'full_native': (.02, .003),
    'gaussian_native': (.02, .003),
    'entry_second_seed': (.0005, .01),
    'entry_previous_level': (.0005, .01),
    'frozen_entry_closure': (.0005, .01),
}
PRODUCTION_SOURCES = (
    'scripts/v3_2_structure_distance.py',
    'docs/v3_2_structure_protocol.md',
    'scripts/v3_2_radial_checkpoint.py',
    'src/photdis/v3/selection.py',
    'src/photdis/v3_1/structure.py',
    'src/photdis/v3_1/selection_dust.py',
    'src/photdis/v3_1/gaussian_response.py',
)


def digest(raw):
    return sha256(raw).hexdigest()


def frozen_parent():
    base = ROOT/'manifests/v3_1/structure'
    parent = digest((base/'matrices/complete.json').read_bytes())
    receipt = json.loads((base/'posterior/complete.json').read_bytes())
    table = (base/'posterior/membership.csv').read_bytes()
    if (receipt['inputs']['data/v3_1/structure/matrices/complete.json'] != parent
            or receipt['outputs']['membership.csv'] != digest(table)):
        raise ValueError('Frozen NGC288 membership roster is not bound to the parent matrices')
    roster = [int(row['identity']) for row in csv.DictReader(io.StringIO(table.decode()))
              if row['branch'] == 'full' and row['TYPE'] == 'PSF']
    if len(roster) != 512 or len(set(roster)) != 512:
        raise ValueError('Incomplete frozen NGC288 PSF membership roster')
    return parent, dict(enumerate(roster))


def read_state(folder, draw, parent, *, pilot=False):
    names = COMPACT_FILES[:2] if pilot else COMPACT_FILES
    payload = {name: (folder/name).read_bytes() for name in names}
    return validate_state(payload, draw, parent, pilot=pilot)


def validate_production_inputs(inputs, draw, parent):
    matrix = 'data/v3_1/structure/matrices/complete.json'
    native = f'data/v3_1/structure/native_refined/draw_{draw:02}/complete.json'
    native_raw = (ROOT/native.replace('data/', 'manifests/', 1)).read_bytes()
    matrix_path = ROOT/matrix.replace('data/', 'manifests/', 1)
    native_sha = digest(native_raw)
    if json.loads(matrix_path.read_bytes())['inputs'][native] != native_sha:
        raise ValueError(f'Frozen matrix references a different native receipt for draw {draw}')
    native_inputs = json.loads(native_raw)['inputs']
    expected = {matrix: parent, native: native_sha}
    for source in ('data/v3_1/structure/photometry.npz',
                   f'data/v3_0/uncertainty_refit/draw_{draw:02}/model.joblib',
                   'data/v3_0/uncertainty/shared_draws.npz'):
        expected[source] = native_inputs[source]
    for source in PRODUCTION_SOURCES:
        expected[source] = digest((ROOT/source).read_bytes())
    if inputs != expected:
        raise ValueError(f'Unexpected production inputs for draw {draw}')


def validate_state(payload, draw, parent, *, pilot=False):
    raw = payload['complete.json']
    receipt = json.loads(raw)
    table = payload['checks.csv']
    n = 3 if pilot else 512
    if ((receipt['draw'], receipt['n'], receipt['new_nodes']) != (draw, n, 384)
            or receipt['pilot'] is not pilot or receipt['all_PSF'] is not True
            or receipt['inputs']['data/v3_1/structure/matrices/complete.json'] != parent
            or receipt['outputs']['checks.csv'] != digest(table)
            or not 0 <= receipt['archived_native_maximum_delta'] <= 1e-8):
        raise ValueError(f'Invalid distance-node receipt for draw {draw}')
    if not pilot:
        validate_production_inputs(receipt['inputs'], draw, parent)
        checkpoint = payload['inputs.json']
        if (digest(checkpoint) != receipt['row_checkpoint_sha256']
                or json.loads(checkpoint) != dict(draw=draw, new_nodes=384, inputs=receipt['inputs'])):
            raise ValueError(f'Invalid distance-node checkpoint for draw {draw}')
    contexts, failures = {}, dict.fromkeys(TOLERANCES, 0)
    for check in csv.DictReader(io.StringIO(table.decode())):
        row, identity, kind = int(check['row']), int(check['identity']), check['check']
        errors = tuple(float(check[k]) for k in ('maximum_absolute_delta', 'maximum_secondary_delta'))
        passed = all(0 <= v < limit for v, limit in zip(errors, TOLERANCES[kind]))
        previous, seen = contexts.setdefault(row, (identity, set()))
        if (not 0 <= row < 512 or check['TYPE'] != 'PSF' or identity != previous or kind in seen
                or check['passed'] != str(passed) or any(not isfinite(v) or v < 0 for v in errors)):
            raise ValueError(f'Invalid context {row}/{kind} in draw {draw}')
        seen.add(kind)
        failures[kind] += not passed
    if (len(contexts) != n or len({i for i, _ in contexts.values()}) != n
            or any(seen != TOLERANCES.keys() for _, seen in contexts.values())
            or sum(failures.values()) != receipt['failures']
            or receipt['passed'] is not (not any(failures.values()))):
        raise ValueError(f'Incomplete context inventory for draw {draw}')
    rows = {row: identity for row, (identity, _) in contexts.items()}
    return dict(draw=draw, job=receipt['job'], n=n, receipt_sha256=digest(raw), failures=failures), rows


def audit(*, complete=False, _recover_partial=False):
    parent, roster = frozen_parent()
    pilot, pilot_rows = read_state(BASE/'pilot', 12, parent, pilot=True)
    if pilot_rows != {i: roster[i] for i in (224, 273, 436)}:
        raise ValueError('Unexpected pilot contexts')
    records = []
    for draw in range(32):
        folder = BASE/f'draw_{draw:02}'
        if not complete and not folder.exists():
            continue
        if _recover_partial and not all((folder/name).is_file() for name in COMPACT_FILES):
            continue
        record, rows = read_state(folder, draw, parent)
        if rows != roster:
            raise ValueError(f'Changed NGC288 parent identity in response state {draw}')
        records.append(record)
    return dict(registered=32, verified=len(records), records=records, pilot=pilot,
                scope='Compact receipts and context verdicts. Native payload closure and joint '
                      'posterior checks run in the allocated science jobs.')


def collect_completed():
    jobs = discover('61263907')
    current = audit(_recover_partial=True)
    archived = {row['draw'] for row in current['records']}
    draws = sorted((jobs.keys() & set(range(32))) - archived)
    if not draws:
        return audit()
    parent, roster = frozen_parent()
    for offset in range(0, len(draws), 8):
        batch = {}
        for draw in draws[offset:offset + 8]:
            folder = f'data/v3_2/structure_distance/draw_{draw:02}'
            batch[draw] = {
                'complete.json': f'{folder}/complete.json',
                'checks.csv': f'{folder}/checks.csv',
                'inputs.json': f'data/v3_2/structure_distance/row_checkpoints/draw_{draw:02}/inputs.json',
            }
        outputs = collect([path for paths in batch.values() for path in paths.values()],
                          max_file_bytes=512_000)
        for draw, paths in batch.items():
            payload = {name: outputs[path] for name, path in paths.items()}
            record, rows = validate_state(payload, draw, parent)
            if record['job'] != jobs[draw] or rows != roster:
                raise ValueError(f'Slurm or parent identity mismatch for draw {draw}')
            archive_verified(BASE/f'draw_{draw:02}', payload)
    return audit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--complete', action='store_true', help='Require the full 32-state inventory')
    mode.add_argument('--collect-completed', action='store_true', help='Archive new successful Slurm states')
    args = parser.parse_args()
    print(json.dumps(collect_completed() if args.collect_completed else audit(complete=args.complete),
                     indent=2))

#!/usr/bin/env python3
"""Verify compact NGC288 distance-node receipts before joint interpretation."""
import argparse
import csv
from hashlib import sha256
import io
import json
from math import isfinite
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v3_2/structure_distance'
TOLERANCES = {
    'full_native': (.02, .003),
    'gaussian_native': (.02, .003),
    'entry_second_seed': (.0005, .01),
    'entry_previous_level': (.0005, .01),
    'frozen_entry_closure': (.0005, .01),
}


def digest(raw):
    return sha256(raw).hexdigest()


def read_state(folder, draw, parent, *, pilot=False):
    raw = (folder/'complete.json').read_bytes()
    receipt = json.loads(raw)
    table = (folder/'checks.csv').read_bytes()
    n = 3 if pilot else 512
    if ((receipt['draw'], receipt['n'], receipt['new_nodes']) != (draw, n, 384)
            or receipt['pilot'] is not pilot or receipt['all_PSF'] is not True
            or receipt['inputs']['data/v3_1/structure/matrices/complete.json'] != parent
            or receipt['outputs']['checks.csv'] != digest(table)
            or not 0 <= receipt['archived_native_maximum_delta'] <= 1e-8):
        raise ValueError(f'Invalid distance-node receipt: {folder}')
    if not pilot:
        checkpoint = (folder/'inputs.json').read_bytes()
        if (digest(checkpoint) != receipt['row_checkpoint_sha256']
                or json.loads(checkpoint) != dict(draw=draw, new_nodes=384, inputs=receipt['inputs'])):
            raise ValueError(f'Invalid distance-node checkpoint: {folder}')
    contexts, failures = {}, dict.fromkeys(TOLERANCES, 0)
    for check in csv.DictReader(io.StringIO(table.decode())):
        row, identity, kind = int(check['row']), int(check['identity']), check['check']
        errors = tuple(float(check[k]) for k in ('maximum_absolute_delta', 'maximum_secondary_delta'))
        passed = all(0 <= v < limit for v, limit in zip(errors, TOLERANCES[kind]))
        previous, seen = contexts.setdefault(row, (identity, set()))
        if (not 0 <= row < 512 or check['TYPE'] != 'PSF' or identity != previous or kind in seen
                or check['passed'] != str(passed) or any(not isfinite(v) or v < 0 for v in errors)):
            raise ValueError(f'Invalid context {row}/{kind}: {folder}')
        seen.add(kind)
        failures[kind] += not passed
    if (len(contexts) != n or len({i for i, _ in contexts.values()}) != n
            or any(seen != TOLERANCES.keys() for _, seen in contexts.values())
            or sum(failures.values()) != receipt['failures']
            or receipt['passed'] is not (not any(failures.values()))):
        raise ValueError(f'Incomplete context inventory: {folder}')
    rows = {row: identity for row, (identity, _) in contexts.items()}
    return dict(draw=draw, job=receipt['job'], n=n, receipt_sha256=digest(raw), failures=failures), rows


def audit(*, complete=False):
    parent = digest((ROOT/'manifests/v3_1/structure/matrices/complete.json').read_bytes())
    pilot, pilot_rows = read_state(BASE/'pilot', 12, parent, pilot=True)
    if set(pilot_rows) != {224, 273, 436}:
        raise ValueError('Unexpected pilot contexts')
    previous, records = None, []
    for draw in range(32):
        folder = BASE/f'draw_{draw:02}'
        if not complete and not folder.exists():
            continue
        record, rows = read_state(folder, draw, parent)
        if ((previous is not None and rows != previous)
                or any(rows[row] != identity for row, identity in pilot_rows.items())):
            raise ValueError(f'Changed NGC288 parent identity in response state {draw}')
        records.append(record)
        previous = rows
    return dict(registered=32, verified=len(records), records=records, pilot=pilot,
                scope='Compact receipts and context verdicts. Native payload closure and joint '
                      'posterior checks run in the allocated science jobs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', action='store_true', help='Require the full 32-state inventory')
    print(json.dumps(audit(complete=parser.parse_args().complete), indent=2))

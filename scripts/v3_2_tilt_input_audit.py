#!/usr/bin/env python3
"""Independently reconcile the compact v3.2 input and refinement receipts."""
import argparse
import csv
from hashlib import sha256
import io
import json
from math import isfinite
from pathlib import Path

from v3_2_tilt_inputs import LEVELS, SHARDS, TILTS
from v3_2_tilt_refine import LEVELS as REFINED_LEVELS

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'manifests/v3_2/tilt_inputs'
OBJECTS = 26000
CANDIDATES = 3540
ABSOLUTE = ('v31_alpha_delta', 'max_tilt_level_delta', 'max_tilt_seed_delta')


def read_receipt(directory, sources):
    marker = directory/'complete.json'
    checks = directory/'checks.csv'
    raw, table = marker.read_bytes(), checks.read_bytes()
    digest = sha256(raw).hexdigest()
    record = json.loads(raw)
    checksum = sha256(table).hexdigest()
    if record['outputs']['checks.csv'] != checksum or record['all_PSF'] is not True:
        raise ValueError(f'Invalid compact receipt: {directory}')
    sources.update({str(marker.relative_to(ROOT)): digest, str(checks.relative_to(ROOT)): checksum})
    return record, list(csv.DictReader(io.StringIO(table.decode()))), digest


def contexts(rows, *, original):
    result = {}
    for r in rows:
        key = (int(r['row']), int(r['identity']), r['admission'])
        errors = [float(r[name]) for name in (*ABSOLUTE, 'stabilized_relative_delta')]
        passed = all(0 <= x < bound for x, bound in zip(errors, (.0005, .0005, .0005, .01)))
        if (r['TYPE'] != 'PSF' or key in result or r['passed'] != str(passed)
                or any(not isfinite(x) or x < 0 for x in errors)
                or (original and not 0 <= float(r['native_log_delta']) <= 1e-8)):
            raise ValueError(f'Invalid check row: {key}')
        result[key] = passed
    return result


def audit(*, complete=False):
    original, refined, unresolved, ids, sources, summary = 0, 0, 0, set(), {}, []
    for name in ('v3_2_tilt_input_audit.py', 'v3_2_tilt_inputs.py', 'v3_2_tilt_refine.py'):
        source = ROOT/'scripts'/name
        sources[str(source.relative_to(ROOT))] = sha256(source.read_bytes()).hexdigest()
    candidates = 0
    for shard in range(SHARDS):
        base = AUDIT/f'shard_{shard:02}'
        patch = AUDIT/f'refined_shard_{shard:02}'
        if not base.exists():
            if complete or patch.exists():
                raise ValueError(f'Missing initial receipt: {shard}')
            continue
        marker, rows, digest = read_receipt(base, sources)
        initial = contexts(rows, original=True)
        broad = {row: identity for row, identity, scope in initial if scope == 'broad'}
        candidate = {row: identity for row, identity, scope in initial if scope == 'candidate'}
        flagged = {k for k, passed in initial.items() if not passed}
        if (marker['shard'] != shard or marker['shards'] != SHARDS
                or marker['levels'] != [list(level) for level in LEVELS] or marker['tilts'] != TILTS.tolist()
                or sorted(broad) != list(range(shard, OBJECTS, SHARDS))
                or len(set(broad.values())) != len(broad) or ids.intersection(broad.values())
                or any(broad.get(row) != identity for row, identity in candidate.items())
                or len(initial) != len(broad)+len(candidate)
                or (marker['n'], marker['candidate_n'], marker['contexts'], marker['failed'])
                != (len(broad), len(candidate), len(initial), len(flagged))):
            raise ValueError(f'Mismatched initial shard: {shard}')
        ids.update(broad.values())
        original += len(flagged)
        candidates += len(candidate)
        status = dict(shard=shard, broad=len(broad), candidate=len(candidate), initial_flags=len(flagged),
                      refinement_present=patch.exists())
        if patch.exists():
            receipt, checks, _ = read_receipt(patch, sources)
            final = contexts(checks, original=False)
            remaining = sum(not passed for passed in final.values())
            if (receipt['shard'] != shard or receipt['source_marker_sha256'] != digest
                    or receipt['levels'] != [list(level) for level in REFINED_LEVELS]
                    or receipt['tilt_probes'] != TILTS.tolist() or set(final) != flagged
                    or receipt['initial_flags'] != len(flagged) or receipt['remaining_flags'] != remaining):
                raise ValueError(f'Mismatched refined shard: {shard}')
            status['remaining_flags'] = remaining
            refined += len(final)
            unresolved += remaining
        elif complete:
            raise ValueError(f'Missing registered refinement receipt: {shard}')
        summary.append(status)
    if complete and (len(ids), candidates) != (OBJECTS, CANDIDATES):
        raise ValueError('Completed census does not match the frozen population sizes')
    return dict(broad=len(ids), candidate=candidates, initial_flags=original, reviewed_flags=refined,
                remaining_flags=unresolved, pending_flags=original-refined, shards=summary, inputs=sources,
                scope='Compact receipt and verdict audit. The allocated fit separately verifies remote arrays.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', action='store_true', help='Require all registered shards and refinements')
    args = parser.parse_args()
    print(json.dumps(audit(complete=args.complete), indent=2))


if __name__ == '__main__':
    main()

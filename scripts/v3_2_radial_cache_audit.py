#!/usr/bin/env python3
"""Verify compact radial-cache receipts against the frozen simulation input."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v3_2'


def digest(raw):
    return sha256(raw).hexdigest()


def audit(base=BASE, *, complete=False):
    design_raw = (base/'radial_design/complete.json').read_bytes()
    design = json.loads(design_raw)
    table = (base/'radial_design/design.csv').read_bytes()
    if digest(table) != design['outputs']['design.csv'] or design['all_PSF'] is not True:
        raise ValueError('Invalid frozen radial design')
    source = {}
    for row in csv.DictReader(table.decode().splitlines()):
        source.setdefault(int(row['repeat']), int(row['catalog']))
    if sorted(source) != list(range(32)):
        raise ValueError('Expected 32 frozen context repeats')
    cache = base/'radial_entry'
    equivalence = json.loads((base/'radial_entry_verification/complete.json').read_text())
    original = cache/'repeat_00/complete.json'
    original_raw = original.read_bytes()
    key = 'data/v3_2/radial_entry/repeat_00/complete.json'
    if equivalence['passed'] is not True or equivalence['inputs'][key] != digest(original_raw):
        raise ValueError('Repeat 00 does not match the exact direct/cache verification')
    signature = json.loads(original_raw)['entry_code_sha256']
    rows = []
    for repeat, catalog in source.items():
        folder = cache/f'repeat_{repeat:02}'
        if not folder.exists() and not complete:
            continue
        raw = (folder/'complete.json').read_bytes()
        checkpoint = (folder/'inputs.json').read_bytes()
        receipt = json.loads(raw)
        expected = dict(repeat=repeat, source_catalog=catalog, n_contexts=128, all_PSF=True,
                        design_sha256=digest(design_raw),
                        model_sha256=design['inputs']['data/v3_0/sed_fit.joblib'],
                        photo_sha256=design['outputs'][f'photometry_{catalog:03}.npz'],
                        entry_code_sha256=signature)
        if (json.loads(checkpoint) != expected or any(receipt[k] != v for k, v in expected.items())
                or receipt['row_checkpoint_sha256'] != digest(checkpoint)
                or set(receipt['outputs']) != {'curves.npz'}):
            raise ValueError(f'Invalid radial cache receipt for repeat {repeat:02}')
        rows.append(dict(repeat=repeat, catalog=catalog, job=receipt['job'],
                         receipt_sha256=digest(raw), curve_sha256=receipt['outputs']['curves.npz']))
    return dict(registered=32, verified=len(rows), contexts=128*len(rows), records=rows,
                scope='Compact receipt and checkpoint inventory. The allocated recovery verifies curve payloads.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', action='store_true')
    print(json.dumps(audit(complete=parser.parse_args().complete), indent=2))

#!/usr/bin/env python3
"""Explain DR19 native photometric exclusions without loading reference labels."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.photometry import derive_photometry
from photdis.stage1 import require_compute, structured
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_0_native import RAW, galactic
from v3_2_tilt_inputs import verified


def entry_components(data):
    raw = structured({k: data[k] for k in RAW if k in data})
    phot = derive_photometry(raw, w2_snr_min=5)
    latitude = galactic(data['RA'], data['DEC'])[1]
    ebv = data['EBV']
    return {
        'psf': data['TYPE'] == 'PSF',
        'grz_base': phot['grz_base'],
        'grz_fraction_clean': phot['grz_fraction_clean'],
        'grz_mask_clean': phot['grz_mask_clean'],
        'high_latitude': abs(latitude) > 30,
        'low_ebv': (ebv >= 0) & (ebv < .03),
    }


def main():
    require_compute()
    root = ROOT/'data/v3_2'
    native = root/'reference_native'
    marker = native/'complete.json'
    receipt = json.loads(marker.read_text())
    if receipt['reference_values_read'] is not False:
        raise ValueError('Photometry-only native receipt required')
    sources = verified(native/'native_sources.json', receipt['outputs']['native_sources.json'])
    products = json.loads(sources.read_text())
    rows = []
    for r in products:
        if not r.get('matched_primary'):
            continue
        path = native/'bricks'/f"{r['brick']}.npz"
        data = load(verified(path, r['sha256']))
        checks = entry_components(data)
        np.testing.assert_array_equal(np.logical_and.reduce(list(checks.values())), data['photometric_entry'])
        for i in range(len(data['TYPE'])):
            rows.append(dict(brick=r['brick'], photometric_entry=bool(data['photometric_entry'][i]),
                             **{name: bool(mask[i]) for name, mask in checks.items()}))
    if len(rows) != receipt['exact_primary_rows']:
        raise ValueError('Native primary-row count mismatch')
    out = root/'reference_entry_audit'
    out.mkdir(exist_ok=False)
    table = out/'gate_patterns.csv'
    csv_rows(table, rows)
    inputs = (marker, sources, Path(__file__).resolve(), ROOT/'scripts/v3_0_native.py',
              ROOT/'src/photdis/photometry.py', ROOT/'src/photdis/stage1.py')
    save_json(out/'complete.json', dict(
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        outputs={table.name: sha(table)}, n=len(rows), reference_values_read=False,
        photometric_entry=sum(r['photometric_entry'] for r in rows),
        job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()

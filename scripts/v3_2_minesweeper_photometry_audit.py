#!/usr/bin/env python3
"""Verify native DR20 pointers and describe the admitted photometry."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.data import SIGNIFICANT, WEAK
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_2_reference_freeze import aligned, require_native, verify_request_pointers
from v3_2_tilt_inputs import verified


def occupancy(photo):
    r0 = photo['r0']
    if not np.isfinite(r0).all():
        raise ValueError('Finite native extinction-corrected r magnitude required')
    states = photo['band_state'][:, 4]
    groups = {
        'weak': states == WEAK,
        'significant': states == SIGNIFICANT,
        'unavailable': ~np.isin(states, [WEAK, SIGNIFICANT]),
    }
    edges = (-np.inf, 18., 20., 22., np.inf)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for name, mask in groups.items():
            selected = mask & (r0 >= lo) & (r0 < hi)
            rows.append(dict(r0_lo=lo, r0_hi=hi, w1=name, n=int(selected.sum()),
                             sky_cells=len(np.unique(photo['cell'][selected]))))
    return rows


def main():
    require_compute()
    root = ROOT/'data/v3_2'
    native = root/'minesweeper_native'
    inventory = root/'minesweeper_inventory'
    admission_path, census_path = native/'complete.json', inventory/'complete.json'
    admission = json.loads(admission_path.read_text())
    verified(census_path, admission['inputs'][str(census_path)])
    census = json.loads(census_path.read_text())
    if admission['reference_values_read'] is not False or census['outcomes_read'] is not False:
        raise ValueError('Outcome-closed native census required')
    photo_path, index_path = (verified(native/name, admission['outputs'][name])
                              for name in ('photometry.npz', 'reference_index.npz'))
    request_path = verified(inventory/'requests.npz', census['outputs']['requests.npz'])
    verified(request_path, admission['inputs'][str(request_path)])
    photo, index, requests = map(load, (photo_path, index_path, request_path))
    require_native(photo)
    aligned(photo, index)
    verify_request_pointers(index, requests, ('reference_row', 'obs'))
    if len(photo['identity']) != admission['admitted'] or not (index['obs'] == 'LCO').all():
        raise ValueError('DR20 cohort differs from its LCO admission')
    rows = occupancy(photo)
    if sum(row['n'] for row in rows) != len(photo['identity']):
        raise ValueError('Photometric strata do not conserve the cohort')
    out = root/'minesweeper_photometry_audit'
    out.mkdir(exist_ok=False)
    table = out/'occupancy.csv'
    csv_rows(table, rows)
    sources = (admission_path, census_path, photo_path, index_path, request_path,
               Path(__file__).resolve(), ROOT/'scripts/v3_2_reference_freeze.py',
               ROOT/'src/photdis/v3/data.py')
    save_json(out/'complete.json', dict(
        inputs={str(p.relative_to(ROOT)): sha(p) for p in sources},
        outputs={table.name: sha(table)}, n=len(photo['identity']),
        sky_cells=len(np.unique(photo['cell'])), all_PSF=True,
        reference_values_read=False, job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()

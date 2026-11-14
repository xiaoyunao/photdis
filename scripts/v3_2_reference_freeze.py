#!/usr/bin/env python3
"""Bind new DR19 reference rows and predictions before outcome access."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3_1.freeze import Inventory, verify_freeze
from v2_1_near import ROOT, load, sha
from v3_2_reference_inventory import ARCHIVES

VIEWS = ('candidate', 'qrf', 'nominal_reference', 'old_target')
IDENTITY = ('identity', 'ls_id', 'TYPE')
ROLE = 'new_dr19_new_sky_field'


def aligned(source, other):
    for name in IDENTITY:
        if name in ('identity', 'ls_id') and other[name].dtype.kind not in 'iu':
            raise ValueError('Exact integer IDs required')
        np.testing.assert_array_equal(other[name], source[name], err_msg=name)


def require_native(photo):
    ids = photo['identity']
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) == 0
            or (ids <= 0).any() or len(np.unique(ids)) != len(ids)
            or photo['ls_id'].dtype.kind not in 'iu' or (photo['ls_id'] <= 0).any()
            or any(photo[key].shape != ids.shape for key in ('ls_id', 'TYPE', 'RELEASE', 'photometric_entry'))
            or len(np.unique(photo['ls_id'])) != len(ids)
            or not (photo['TYPE'] == 'PSF').all() or not (photo['RELEASE'] == 11010).all()
            or not photo['photometric_entry'].all()):
        raise ValueError('Nonempty distinct verified DR11 PSF identities required')


def read_admission(path, photo, expected_n):
    with np.load(path) as data:
        aligned(photo, data)
        gate = data['candidate']
        if gate.dtype.kind != 'b' or gate.shape != photo['identity'].shape or gate.sum() != expected_n:
            raise ValueError('Invalid matched candidate admission mask')
        return gate


def bind_predictions(inv, marker, photo, photo_digest, decision_digest):
    record = inv.marker(marker)
    if (record.get('references_read') is not False or record.get('pilot') is not False
            or record.get('cohort') != ROLE
            or record.get('photometry_sha256') != photo_digest
            or record.get('decision_sha256') != decision_digest
            or set(record['views']) != set(VIEWS)):
        raise ValueError('A full photometry-only forecast from all frozen views is required')
    photo = {k: photo[k] for k in IDENTITY}
    admission = record['candidate_admission']
    read_admission(inv.add(admission['payload'], admission['sha256']), photo, admission['admitted'])
    for name in VIEWS:
        view = record['views'][name]
        view_path = inv.add(view['payload'], view['sha256'])
        with np.load(view_path) as data:
            aligned(photo, data)
            n = len(photo['identity'])
            if data['mu_quantiles'].shape != (n, 5):
                raise ValueError(f'Missing distance quantiles: {name}')
            if name == 'qrf':
                available = data['available']
                if (available.shape != (n,) or available.dtype.kind != 'b'
                        or data['mass'].ndim != 2 or data['mass'].shape[0] != n
                        or data['mu_edges'].shape != (n, data['mass'].shape[1]+1)):
                    raise ValueError('Invalid native QRF forecast arrays')
            else:
                for prefix in ('coarse_', ''):
                    grid = data[prefix+'mu_grid']
                    if (grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all()
                            or not (np.diff(grid) > 0).all()
                            or data[prefix+'logpdf'].shape != (n, len(grid))
                            or data[prefix+'mu_quantiles'].shape != (n, 5)
                            or data[prefix+'class_distance'].shape != (n, 3, 6)):
                        raise ValueError(f'Invalid continuous forecast arrays: {name} {prefix}')
            status = data['numerical_pass']
            if status.shape != photo['identity'].shape or status.dtype.kind != 'b':
                raise ValueError(f'Missing per-object numerical status: {name}')


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('decision', help='Frozen mechanism and opened evaluation decision marker')
    parser.add_argument('forecast', help='Complete photometry-only four-view forecast marker')
    parser.add_argument('evaluator', help='Registered scoring program for the frozen views')
    args = parser.parse_args()
    root = ROOT/'data/v3_2'
    out = root/'final_validation_dr19'
    out.mkdir(parents=True, exist_ok=True)
    output = out/'freeze.json'
    if output.exists():
        print('EXISTING FREEZE', verify_freeze(ROOT, output), flush=True)
        return
    if (out/'started.json').exists():
        raise ValueError('Outcomes already accessed for this role')
    inv = Inventory(ROOT)
    native = root/'reference_native'
    admitted = inv.marker(native/'complete.json')
    if admitted.get('reference_values_read') is not False or not admitted['all_admitted_PSF']:
        raise ValueError('Require outcome-closed PSF admission')
    photo_path = inv.add(native/'photometry.npz', admitted['outputs']['photometry.npz'])
    index_path = inv.add(native/'reference_index.npz', admitted['outputs']['reference_index.npz'])
    photo, index = load(photo_path), load(index_path)
    require_native(photo)
    aligned(photo, index)
    for key in ('sdss_id', 'distance_row', 'stellar_row'):
        v = index[key]
        if v.shape != photo['identity'].shape or v.dtype.kind not in 'iu' or (v < 0).any() or len(np.unique(v)) != len(v):
            raise ValueError(f'Unique integer reference pointers required: {key}')
    inventory = inv.marker(root/'reference_inventory/complete.json')
    if inventory.get('outcomes_read') is not False:
        raise ValueError('Reference outcomes were read during identity selection')
    for archive in ARCHIVES.values():
        path = root/'reference_sources/dr19_astra_0.6.0'/archive
        inv.add(path, inventory['inputs'][str(path)])
    decision_path = inv.add(args.decision)
    decision = inv.marker(decision_path)
    if decision.get('development_evaluation_complete') is not True or decision.get('new_outcomes_read') is not False:
        raise ValueError('Require a completed opened-domain mechanism decision')
    forecast_path = inv.add(args.forecast)
    bind_predictions(inv, forecast_path, photo, sha(photo_path), sha(decision_path))
    evaluator = inv.add(args.evaluator)
    for p in ('docs/v3_2_reference_intake.md', 'scripts/v3_2_reference_freeze.py',
              'scripts/v3_2_reference_evidence.py', 'src/photdis/v3_1/freeze.py', 'src/photdis/gaia.py',
              'src/photdis/v3/metrics.py', 'src/photdis/v3_1/validation_report.py',
              'src/photdis/v3_1/validation_scores.py'):
        inv.add(p)
    inv.unchanged()
    receipt = dict(cohort=ROLE, n=len(photo['identity']), all_PSF=True,
                   predictions_frozen=True, outcomes_opened=False, views=list(VIEWS),
                   decision=str(decision_path.relative_to(ROOT)),
                   forecast=str(forecast_path.relative_to(ROOT)),
                   evaluator=str(evaluator.relative_to(ROOT)),
                   hashes=inv.hashes, job=os.getenv('SLURM_JOB_ID'),
                   utc=datetime.now(timezone.utc).isoformat())
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()

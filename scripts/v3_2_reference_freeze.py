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
MINESWEEPER_ROLE = 'new_dr20_lco_new_sky_field'
COMMON_ROLE = 'new_hst_kinematic_system'


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


def check_quantiles(values, n, name):
    if values.shape != (n, 5) or not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise ValueError(f'Invalid distance quantiles: {name}')


def bind_predictions(inv, marker, photo, photo_digest, decision_digest, *, role=ROLE):
    record = inv.marker(marker)
    if (record.get('references_read') is not False or record.get('pilot') is not False
            or record.get('cohort') != role
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
                edges, mass = data['mu_edges'][available], data['mass'][available]
                check_quantiles(data['mu_quantiles'][available], int(available.sum()), name)
                if (not np.isfinite(edges).all() or not (np.diff(edges, axis=1) > 0).all()
                        or not np.isfinite(mass).all() or (mass < 0).any()
                        or not np.allclose(mass.sum(axis=1), 1.)):
                    raise ValueError('Invalid available QRF histogram')
            else:
                for prefix in ('coarse_', ''):
                    grid = data[prefix+'mu_grid']
                    logpdf = data[prefix+'logpdf']
                    probability = data[prefix+'class_distance']
                    if (grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all()
                            or not (np.diff(grid) > 0).all()
                            or logpdf.shape != (n, len(grid)) or probability.shape != (n, 3, 6)):
                        raise ValueError(f'Invalid continuous forecast arrays: {name} {prefix}')
                    check_quantiles(data[prefix+'mu_quantiles'], n, name)
                    finite = np.isfinite(logpdf)
                    starts = finite[:, 0].astype(int) + ((~finite[:, :-1]) & finite[:, 1:]).sum(axis=1)
                    if (np.isnan(logpdf).any() or np.isposinf(logpdf).any()
                            or (finite.sum(axis=1) < 3).any() or (starts != 1).any()
                            or not np.isfinite(probability).all() or (probability < 0).any()
                            or (abs(probability.sum(axis=(1, 2))-1.) > 1e-8).any()):
                        raise ValueError(f'Invalid continuous probability: {name} {prefix}')
            status = data['numerical_pass']
            if status.shape != photo['identity'].shape or status.dtype.kind != 'b':
                raise ValueError(f'Missing per-object numerical status: {name}')


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('decision', help='Frozen mechanism and opened evaluation decision marker')
    parser.add_argument('forecast', help='Complete photometry-only four-view forecast marker')
    parser.add_argument('evaluator', help='Registered scoring program for the frozen views')
    parser.add_argument("--minesweeper", action="store_true")
    args = parser.parse_args()
    root = ROOT/'data/v3_2'
    role = MINESWEEPER_ROLE if args.minesweeper else ROLE
    out = root/('final_validation_dr20' if args.minesweeper else 'final_validation_dr19')
    out.mkdir(parents=True, exist_ok=True)
    output = out/'freeze.json'
    if output.exists():
        digest = verify_freeze(ROOT, output)
        saved = json.loads(output.read_text())
        if saved['cohort'] != role:
            raise ValueError('Existing freeze has the wrong cohort')
        for key in ('decision', 'forecast', 'evaluator'):
            if (ROOT/saved[key]).resolve() != Path(getattr(args, key)).resolve():
                raise ValueError(f'Existing freeze binds a different {key}')
        print('EXISTING FREEZE', digest, flush=True)
        return
    if (out/'started.json').exists():
        raise ValueError('Outcomes already accessed for this role')
    inv = Inventory(ROOT)
    prefix = 'minesweeper' if args.minesweeper else 'reference'
    native = root/f'{prefix}_native'
    admitted = inv.marker(native/'complete.json')
    if admitted.get('reference_values_read') is not False or not admitted['all_admitted_PSF']:
        raise ValueError('Require outcome-closed PSF admission')
    photo_path = inv.add(native/'photometry.npz', admitted['outputs']['photometry.npz'])
    index_path = inv.add(native/'reference_index.npz', admitted['outputs']['reference_index.npz'])
    photo, index = load(photo_path), load(index_path)
    require_native(photo)
    aligned(photo, index)
    if args.minesweeper and not (index['obs'] == 'LCO').all():
        raise ValueError('DR20 freeze requires LCO pointers')
    pointer_keys = ('reference_row',) if args.minesweeper else ('sdss_id', 'distance_row', 'stellar_row')
    for key in pointer_keys:
        v = index[key]
        if v.shape != photo['identity'].shape or v.dtype.kind not in 'iu' or (v < 0).any() or len(np.unique(v)) != len(v):
            raise ValueError(f'Unique integer reference pointers required: {key}')
    inventory = inv.marker(root/f'{prefix}_inventory'/'complete.json')
    if inventory.get('outcomes_read') is not False:
        raise ValueError('Reference outcomes were read during identity selection')
    if admitted['inputs'][str(root/f'{prefix}_inventory'/'complete.json')] != sha(root/f'{prefix}_inventory'/'complete.json'):
        raise ValueError('Native admission binds a different identity census')
    archives = (root/'reference_sources/dr20_minesweeper/minesweeper_v1.2.2.fits',) if args.minesweeper else (
        tuple(root/'reference_sources/dr19_astra_0.6.0'/name for name in ARCHIVES.values()))
    for path in archives:
        inv.add(path, inventory['inputs'][str(path)])
    if args.minesweeper:
        inv.add('docs/v3_2_minesweeper_intake.md')
    decision_path = inv.add(args.decision)
    decision = inv.marker(decision_path)
    if decision.get('development_evaluation_complete') is not True or decision.get('new_outcomes_read') is not False:
        raise ValueError('Require a completed opened-domain mechanism decision')
    forecast_path = inv.add(args.forecast)
    bind_predictions(inv, forecast_path, photo, sha(photo_path), sha(decision_path), role=role)
    evaluator = inv.add(args.evaluator)
    for p in ('docs/v3_2_reference_intake.md', 'scripts/v3_2_reference_freeze.py',
              'scripts/v3_2_reference_evidence.py', 'scripts/v3_2_contamination_support.py',
              'src/photdis/v3_1/freeze.py', 'src/photdis/gaia.py',
              'src/photdis/v3/metrics.py', 'src/photdis/v3_1/validation_report.py',
              'src/photdis/v3_1/validation_scores.py'):
        inv.add(p)
    inv.unchanged()
    receipt = dict(cohort=role, n=len(photo['identity']), all_PSF=True,
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

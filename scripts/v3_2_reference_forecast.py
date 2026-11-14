#!/usr/bin/env python3
"""Assemble authenticated prospective forecasts for the final freeze."""
import argparse
import os
from pathlib import Path

from photdis.stage1 import require_compute
from photdis.v3_1.freeze import Inventory
from v2_1_near import ROOT, load, save_json, sha
from v3_2_reference_freeze import COMMON_ROLE, ROLE, VIEWS, bind_predictions, require_native


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('decision')
    parser.add_argument('candidate', help='Candidate-only forecast completion receipt')
    parser.add_argument('--common', action='store_true', help='Assemble the common-system forecasts')
    args = parser.parse_args()
    root = ROOT/'data/v3_2'
    inv = Inventory(ROOT)
    role = COMMON_ROLE if args.common else ROLE
    native = 'common_scale_inventory' if args.common else 'reference_native'
    controls = 'common_scale_controls' if args.common else 'reference_controls'
    photo_path = root/native/'photometry.npz'
    admission = inv.marker(photo_path.with_name('complete.json'))
    photo = load(inv.add(photo_path, admission['outputs'][photo_path.name]))
    require_native(photo)
    photo_digest = sha(photo_path)
    decision_path = inv.add(args.decision)
    decision_digest = sha(decision_path)
    decision = inv.marker(decision_path)
    if decision.get('development_evaluation_complete') is not True or decision.get('new_outcomes_read') is not False:
        raise ValueError('Register and evaluate the mechanism before assembly')
    views = {}
    for path, expected in [(root/controls/'complete.json', set(VIEWS)-{'candidate'}),
                           (inv.add(args.candidate), {'candidate'})]:
        record = inv.marker(path)
        if (record.get('cohort') != role or record.get('references_read') is not False
                or record.get('pilot') is not False or record['photometry_sha256'] != photo_digest
                or record['n'] != len(photo['identity']) or set(record['views']) != expected):
            raise ValueError(f'Incomplete or conflicting forecast: {path}')
        if 'candidate' in expected:
            if record['decision_sha256'] != decision_digest:
                raise ValueError('Candidate does not bind the completed mechanism decision')
        else:
            candidate_admission = record['candidate_admission']
        views.update(record['views'])
    inv.add(Path(__file__).resolve())
    inv.add(ROOT/'scripts/v3_2_reference_freeze.py')
    receipt = dict(cohort=role, n=len(photo['identity']), all_PSF=True, references_read=False,
                   pilot=False, photometry_sha256=photo_digest, decision_sha256=decision_digest,
                   views=views, candidate_admission=candidate_admission,
                   inputs=inv.hashes.copy(), job=os.getenv('SLURM_JOB_ID'))
    destination = root/('common_scale_forecast' if args.common else 'reference_forecast')
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'complete.json'
    # Validate in staging before creating the canonical completion receipt.
    staging = destination/'checked.json'
    save_json(staging, receipt)
    bind_predictions(inv, staging, photo, photo_digest, decision_digest, role=role)
    inv.unchanged()
    staging.rename(path)


if __name__ == '__main__':
    main()

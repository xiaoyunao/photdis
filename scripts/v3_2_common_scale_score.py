#!/usr/bin/env python3
"""Freeze common-system forecasts and evaluate shared HST scales once."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3_1.freeze import Inventory, verify_freeze
from photdis.v3_1.validation_report import common_scale_statistics
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_2_reference_freeze import COMMON_ROLE, VIEWS, bind_predictions, read_admission, require_native


def shared_distances(distance, lower, upper, system):
    if not np.isfinite([distance, lower, upper]).all() or min(distance, lower, upper) <= 0:
        raise ValueError('Positive finite HST scale and uncertainties required')
    normal = np.random.default_rng(np.random.SeedSequence([320501, system])).normal(size=8192)
    values = distance + normal*np.where(normal < 0, lower, upper)
    if (values > 0).sum() < 4096:
        raise ValueError('Insufficient positive common-scale draws')
    return values[values > 0][:4096]


def metrics(forecasts, photo, candidate, scales):
    rows = []
    for group, (system, distance, lower, upper) in sorted(scales.items()):
        draws = shared_distances(distance, lower, upper, group)
        members = photo['scale_system'] == group
        for scope, mask in [('native_photometric', members),
                            ('matched_candidate_admission', members & candidate)]:
            availability = {view: data.get('available', np.ones(len(mask), bool))
                            for view, data in forecasts.items()}
            for view, data in forecasts.items():
                q = data['mu_quantiles']
                for accepted in (False, True):
                    use = mask & availability[view] & (data['numerical_pass'] if accepted else True)
                    pair = use & availability['candidate']
                    if accepted:
                        pair &= forecasts['candidate']['numerical_pass']
                    for is_pair, selected in ((False, use), (True, pair)):
                        if is_pair and view == 'candidate':
                            continue
                        left = common_scale_statistics(q[selected], draws)
                        right = (common_scale_statistics(forecasts['candidate']['mu_quantiles'][selected], draws)
                                 if is_pair else {})
                        for name, value in left.items():
                            delta = right[name]-value if is_pair else value
                            limits = np.quantile(delta, [.025, .5, .975]).tolist() if selected.any() else [None]*3
                            rows.append(dict(system=system, view=view, candidate_minus_view=is_pair,
                                             scope=scope, numerical_only=accepted, metric=name,
                                             registered=int(mask.sum()), n=int(selected.sum()),
                                             distance_kpc=distance, lower_error_kpc=lower, upper_error_kpc=upper,
                                             lo95=limits[0], median=limits[1], hi95=limits[2]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('decision', help='Completed opened-development mechanism decision')
    args = parser.parse_args()
    require_compute()
    root = ROOT/'data/v3_2'
    native = root/'common_scale_inventory'
    out = root/'final_validation_common'
    out.mkdir(parents=True, exist_ok=True)
    freeze, started, complete = [out/name for name in ('freeze.json', 'started.json', 'complete.json')]
    inv = Inventory(ROOT)
    admission = inv.marker(native/'complete.json')
    if admission.get('scale_values_exported') is not False or not admission['all_admitted_PSF']:
        raise ValueError('Require outcome-closed common-system PSF identities')
    photo_path = inv.add(native/'photometry.npz', admission['outputs']['photometry.npz'])
    photo = load(photo_path)
    require_native(photo)
    groups = np.unique(photo['scale_system'])
    if len(groups) != admission['admitted_systems'] or len(photo['identity']) != admission['admitted_members']:
        raise ValueError('Common-system identity census changed')
    decision = inv.add(args.decision)
    registration = inv.marker(decision)
    if registration.get('development_evaluation_complete') is not True or registration.get('new_outcomes_read') is not False:
        raise ValueError('Require completed opened-development decision')
    forecast = root/'common_scale_forecast/complete.json'
    bind_predictions(inv, forecast, photo, sha(photo_path), sha(decision), role=COMMON_ROLE)
    reference = inv.add('manifests/kgiant_bv21_reference_v1.json',
                        admission['inputs'][str(ROOT/'manifests/kgiant_bv21_reference_v1.json')])
    for source in (Path(__file__).resolve(), 'scripts/v3_2_reference_freeze.py',
                   'docs/v3_2_reference_intake.md', 'src/photdis/v3_1/validation_report.py',
                   'src/photdis/v3_1/freeze.py'):
        inv.add(source)
    inv.unchanged()
    if not freeze.exists():
        if started.exists():
            raise ValueError('Outcome receipt without matching freeze')
        with freeze.open('x') as stream:
            json.dump(dict(cohort=COMMON_ROLE, predictions_frozen=True, outcomes_opened=False,
                           n=len(photo['identity']), hashes=inv.hashes, job=os.getenv('SLURM_JOB_ID')),
                      stream, indent=2)
    digest = verify_freeze(ROOT, freeze)
    if json.loads(freeze.read_text())['hashes'] != inv.hashes:
        raise ValueError('Common-system freeze differs from requested inputs')
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt['freeze_sha256'] != digest or sha(out/'scale_agreement.csv') != receipt['outputs']['scale_agreement.csv']:
            raise ValueError('Common-system result differs from its receipt')
        return
    if started.exists():
        if json.loads(started.read_text())['freeze_sha256'] != digest:
            raise ValueError('Existing outcome receipt binds a different freeze')
    else:
        with started.open('x') as stream:
            json.dump(dict(freeze_sha256=digest, utc=datetime.now(timezone.utc).isoformat(),
                           job=os.getenv('SLURM_JOB_ID')), stream)
    catalog = json.loads(reference.read_text())
    published = {r['name']: r for r in catalog['rows']}
    mapping = {r['group']: r for r in catalog['membership_mapping']}
    scales = {}
    for group in groups:
        row = mapping[group]
        value = published[row['reference_name']]['methods']['hst_kinematic']
        scales[int(group)] = (row['member_name'], value['distance_kpc'], value['minus_kpc'], value['plus_kpc'])
    record = json.loads(forecast.read_text())
    gate = record['candidate_admission']
    candidate = read_admission(ROOT/gate['payload'], photo, gate['admitted'])
    predictions = {view: load(ROOT/record['views'][view]['payload']) for view in VIEWS}
    report = out/'scale_agreement.csv'
    csv_rows(report, metrics(predictions, photo, candidate, scales))
    inv.unchanged()
    save_json(complete, dict(freeze_sha256=digest, outcomes_opened=True, all_PSF=True,
              n=len(photo['identity']), n_systems=len(groups), job=os.getenv('SLURM_JOB_ID'),
              outputs={report.name: sha(report)}))


if __name__ == '__main__':
    main()

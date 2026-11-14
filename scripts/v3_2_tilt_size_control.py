#!/usr/bin/env python3
"""Fit, forecast and score a pre-registered size-matched broad control."""
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3_1.validation_report import paired_field_statistic
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_2_tilt_contrasts import METRICS, matched_rows, number
from v3_2_tilt_fit import main as fit
from v3_2_tilt_inputs import verified
from v3_2_tilt_predict import main as forecast


SIZE = 3540
SEED = 320617
RUN = 'size_control'
ARMS = ('broad_zero', 'broad_free')
REGISTRATION = (Path(__file__).resolve(), ROOT / 'docs/v3_2_tilt_size_protocol.md')


def fixed_sample(data):
    n = len(data['identity'])
    if (n != 26000 or len(np.unique(data['identity'])) != n
            or not (data['TYPE'] == 'PSF').all()):
        raise ValueError('Expected unique verified PSFs from the full broad fit')
    np.testing.assert_array_equal(data['row'], np.arange(n))
    index = np.sort(np.random.Generator(np.random.PCG64(SEED)).choice(n, SIZE, replace=False))
    if not np.array_equal(np.unique(data['field'][index]), np.unique(data['field'])):
        raise ValueError('Size control does not cover all registered broad fields')
    return {key: values[index] for key, values in data.items()}


def read_forecast(run, arms):
    folder = ROOT / 'data/v3_2/tilt_predictions' / run
    receipt = folder / 'complete.json'
    record = json.loads(receipt.read_text())
    if not record['all_PSF'] or record['arms'] != list(arms) or record['n'] != 526:
        raise ValueError(f'Unexpected opened field forecast: {receipt}')
    table = verified(folder / 'objects.csv', record['outputs']['objects.csv'])
    with table.open(newline='') as stream:
        return list(csv.DictReader(stream)), receipt


def score():
    require_compute()
    all_arms = (*ARMS, 'candidate_zero', 'candidate_free')
    initial, baseline_receipt = read_forecast('initial', all_arms)
    controls, control_receipt = read_forecast(RUN, ARMS)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (*REGISTRATION, baseline_receipt, control_receipt)}
    samples = []
    fit_root = ROOT / 'data/v3_2/tilt_fits' / RUN
    for arm in ARMS:
        folder = fit_root / arm
        path = folder / 'complete.json'
        hashes[str(path.relative_to(ROOT))] = sha(path)
        fit_record = json.loads(path.read_text())
        sample = load(verified(folder / 'sample.npz', fit_record['outputs']['sample.npz']))
        if not fit_record['all_PSF'] or fit_record['n'] != SIZE or not (sample['TYPE'] == 'PSF').all():
            raise ValueError(f'Unexpected fit sample for {arm}')
        samples.append(sample)
    for key in samples[0]:
        np.testing.assert_array_equal(samples[0][key], samples[1][key])

    by_key = {(r['identity'], r['view']): r for r in controls}
    if len(by_key) != len(controls) or len(by_key) != 526 * len(ARMS):
        raise ValueError('Require one forecast per object and size-control arm')
    groups = matched_rows(initial)
    fields = ('TYPE', 'role', 'cell', 'class_label', 'candidate_admission')
    for group in groups:
        for arm in ARMS:
            original = group[arm]
            control = by_key.pop((original['identity'], arm))
            if any(control[field] != original[field] for field in fields):
                raise ValueError('Forecast row belongs to a different evaluation context')
            group[arm + '_3540'] = control

    rows = []
    for role in ('new_sky_field', 'new_source_seen_sky_field'):
        admitted = [g for g in groups if g['broad_zero']['role'] == role
                    and g['broad_zero']['candidate_admission'] == 'True']
        for stratum, label in (('all', None), ('dwarf', 0), ('giant', 1)):
            group = [g for g in admitted if label is None or int(g['broad_zero']['class_label']) == label]
            cells = np.array([int(g['broad_zero']['cell']) for g in group])
            for arm in ARMS:
                modes = ((arm + '_3540', arm), (arm.replace('broad', 'candidate'), arm + '_3540'))
                for left, right in modes:
                    for metric in METRICS:
                        a = [number(g[left], metric) for g in group]
                        b = [number(g[right], metric) for g in group]
                        rows.append(dict(role=role, stratum=stratum, left=left, right=right, metric=metric,
                                         group_n=len(group), **paired_field_statistic(a, b, cells)))
    output = ROOT / 'data/v3_2/tilt_predictions' / RUN / 'comparison'
    output.mkdir(exist_ok=False)
    csv_rows(output / 'paired.csv', rows)
    save_json(output / 'complete.json', dict(
        inputs=hashes, outputs={'paired.csv': sha(output / 'paired.csv')},
        job=os.getenv('SLURM_JOB_ID'), sample_psf=SIZE, seed=SEED, n=len(groups),
        all_PSF=True, new_outcomes_read=False,
        grid_flags={arm: sum(r['view'] == arm and r['grid_passed'] != 'True' for r in controls)
                    for arm in ARMS},
        sampler_passed={arm: all(r['sampler_passed'] == 'True' for r in controls if r['view'] == arm)
                        for arm in ARMS},
    ))


if __name__ == '__main__':
    phase = sys.argv.pop(1)
    if phase == 'fit':
        fit(broad_size_control=fixed_sample, registration=REGISTRATION)
    elif phase == 'forecast':
        forecast(run=RUN, arm_names=ARMS, fit_inputs=REGISTRATION)
    elif phase == 'score':
        score()
    else:
        raise ValueError(f'Unknown size-control phase: {phase}')

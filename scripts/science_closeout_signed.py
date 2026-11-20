#!/usr/bin/env python3
"""Field-wise signed-parallax power and conditional scoring of frozen QRF PDFs."""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

import faint_signed_20261120 as prior
from photdis.faint_signed import scale_weights
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha

OUT = ROOT / 'data/science_closeout_20261120/signed'
CONFIG = ROOT / 'configs/science_closeout_20261120.json'


def traits(t, table, stress=1.):
    blocks = np.unique(t['block'], return_inverse=True)[1]
    field_rms, block_rms = (prior.C[k] * stress for k in
                            ('field_offset_rms_mas', 'block_offset_rms_mas'))
    slope = (table[:, 3] - table[:, 1]) / .6
    response = table[:, -1]
    variance = table[:, -3]
    weights = scale_weights(slope, response, variance, blocks, field_rms, block_rms)
    weighted_response = weights * response
    sd = np.sqrt(np.dot(weights**2, variance) + (field_rms * weighted_response.sum())**2
                 + block_rms**2 * np.square(np.bincount(blocks, weights=weighted_response)).sum())
    return blocks, weights, sd


def cohort(field, root, coordinates, magnitude, expected):
    roster = load(root / 'forecast/roster.npz')
    phot = load(root / 'census/qualified.npz')
    np.testing.assert_array_equal(roster['ls_id'], phot['ls_id'])
    if root == prior.TRANSFER:
        masks = load(root / 'preflight/identity_masks.npz')
        np.testing.assert_array_equal(roster['ls_id'], masks['ls_id'])
        independent = masks['clean']
    else:
        independent = ~(roster['qrf_fit_overlap'] | roster['historical_overlap'])
    select = ((roster['field'] == field) & independent & roster['support']
              & (roster['gaia_state'] == 4) & (roster['r'] >= magnitude)
              & (roster['r'] < magnitude + 1))
    all_field = np.flatnonzero((roster['field'] == field) & (roster['source_id'] > 0))
    ids = roster['source_id'][all_field]
    unique, counts = np.unique(ids, return_counts=True)
    select &= np.isin(roster['source_id'], unique[counts == 1])
    rows = np.flatnonzero(select)
    if (len(rows) != int(expected['qrf_history_clean_weak'])
            or np.count_nonzero(roster['gaia_parallax'][rows] < 0)
            != int(expected['qrf_history_clean_negative'])):
        # The raw replay applies the parent-source multiplicity definition.
        return prior.make_cohort(field, root, coordinates, magnitude, True)
    rank = stable_rank(roster['ls_id'][rows], prior.C['seed'])
    rows = rows[np.argsort(rank, kind='stable')[:prior.C['pilot_cap']]]
    source = {k: roster[k][rows] for k in
              ('TYPE', 'ls_id', 'source_id', 'r', 'longitude', 'latitude', 'gaia_parallax_error')}
    source.update({k: phot[k][rows] for k in ('mag_grz', 'error_grz', 'identity')})
    assert (source['TYPE'] == 'PSF').all()
    return source, roster['gaia_parallax'][rows]


def run_field(field, workers):
    import json

    prior.C.update(json.loads(CONFIG.read_text())['signed'])
    prior.OUT = OUT
    prior.traits = traits
    root, position = prior.registered_fields()[field]
    with prior.PREVIOUS.open() as stream:
        targets = {int(row['r_lo']): row for row in csv.DictReader(stream) if row['field'] == field}
    destination = OUT / 'results' / field
    destination.mkdir(parents=True, exist_ok=True)
    power, scoring, scores, gate, pit, snr = [], [], [], [], [], []
    models, _, model_paths = prior.prepare()
    prior.MODEL = models['qrf']
    for magnitude in (20, 21):
        target = targets[magnitude]
        expected = int(target['qrf_history_clean_weak'])
        folder = OUT / 'pilot' / f'{field}_{magnitude}'
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(field=field, r_lo=magnitude, parent_weak=expected, evaluated=0, blocks=0,
                   pass_03=False, pass_05=False, pass_03_double=False, pass_05_double=False)
        print('Begin', field, magnitude, expected, flush=True)
        if expected:
            source, observed = cohort(field, root, position, magnitude, target)
            assert len(observed) == min(expected, prior.C['pilot_cap'])
            assert np.all(np.isfinite(observed) & (observed < 5 * source['gaia_parallax_error']))
            prior.INPUT = source
            source['block'] = prior.spatial.covariance_blocks(source)
            row.update(evaluated=len(observed), blocks=len(np.unique(source['block'])))
            atomic_npz(folder / 'inputs.npz', **{k: source[k] for k in
                       ('TYPE', 'ls_id', 'source_id', 'r', 'block', 'gaia_parallax_error')})
            chunks = np.array_split(np.arange(len(observed)), int(np.ceil(len(observed) / 256)))
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork')) as pool:
                result = list(pool.map(prior.replay_qrf, chunks))
            edges, mass = (np.concatenate(part) for part in zip(*result))
            atomic_npz(folder / 'pdf.npz', ls_id=source['ls_id'], TYPE=source['TYPE'], edges=edges, mass=mass)
            experiments = prior.power_cell((field, magnitude))
            power.extend(experiments)
            lo, hi = prior.C['allowed_realized_null_rejection']
            for stress, suffix in ((1, ''), (2, '_double')):
                runs = {p['shift_mag']: p for p in experiments if p['systematics_multiplier'] == stress}
                for scale, label in ((.3, '03'), (.5, '05')):
                    row['pass_' + label + suffix] = (row['blocks'] >= prior.C['minimum_blocks']
                        and lo <= runs[0]['rejection'] <= hi
                        and min(runs[-scale]['directional_power'], runs[scale]['directional_power'])
                        >= prior.C['minimum_power'])
            # Freeze the decision before opening the observation to the scoring routine.
            save_json(folder / 'gate.json', row)
            atomic_npz(folder / 'signed_observation.npz', ls_id=source['ls_id'],
                       TYPE=source['TYPE'], parallax=observed)
            if row['pass_03'] or row['pass_05']:
                record, comparisons, hist, counts = prior.summarize(folder, 'field_power', field, magnitude)
                scoring.append(record)
                for comparison in comparisons:
                    shift = abs(comparison['shift_mag'])
                    if shift == 0 or (shift == .3 and row['pass_03']) or (shift == .5 and row['pass_05']):
                        scores.append(comparison)
                pit.extend(hist)
                snr.extend(counts)
        gate.append(row)
        print(row, flush=True)
    for name, rows in (('gate', gate), ('power', power), ('signed_summary', scoring),
                       ('signed_scores', scores), ('pit', pit), ('signed_snr', snr)):
        if rows:
            csv_rows(destination / f'{name}.csv', rows)
    save_json(destination / 'complete.json', dict(config=sha(CONFIG), field=field,
        slurm_job=os.environ.get('SLURM_JOB_ID'), model_hashes={str(p): sha(p) for p in model_paths},
        outputs={p.name: sha(p) for p in destination.glob('*.csv')}))


if __name__ == '__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('field')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    run_field(args.field, args.workers)

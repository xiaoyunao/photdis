#!/usr/bin/env python3
"""Qualify identity-matched field photometry without opening reference values."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import fitsio
import numpy as np
from scipy.spatial import cKDTree

from photdis.assembly_v1_4 import unit_vectors, uppercase_fields
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_native import NEW_SKY
from v3_1_external_history import HISTORY, identity_position
import v3_1_external_native as native
from v3_1_freeze_roles import history_columns
from v3_2_reference_inventory import footprint
from v3_2_tilt_inputs import verified


def historical_exclusions(data, paths):
    hit = np.zeros(len(data['identity']), bool)
    rows = []
    for path in paths:
        old = history_columns(path)
        ids = old.get('identity', old.get('source_id'))
        imaging = old['ls_id']
        if ids.dtype.kind not in 'iu' or imaging.dtype.kind not in 'iu':
            raise ValueError('Historical identities must be exact integers')
        overlap = (np.isin(data['identity'], ids[ids > 0])
                   | np.isin(data['ls_id'], imaging[imaging > 0])
                   | np.isin(-data['ls_id'], ids[ids < 0]))
        _, sky = identity_position(path)
        block = np.isin(data['cell'], sky['cell'][sky['valid_coordinates']])
        hit |= overlap | block
        rows.append(dict(path=str(path.relative_to(ROOT)), identity_overlap=int(overlap.sum()),
                         sky_overlap=int(block.sum())))
    return hit, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minesweeper', action='store_true',
                        help='Match the closed DR20 MINESweeper identity census')
    args = parser.parse_args()
    require_compute()
    prefix = 'minesweeper' if args.minesweeper else 'reference'
    root = ROOT/'data/v3_2'
    source = root/f'{prefix}_inventory'
    marker = source/'complete.json'
    record = json.loads(marker.read_text())
    if record['outcomes_read'] or record['scientific_sample_exported']:
        raise ValueError('Require a photometry-only native match request')
    requests = load(verified(source/'requests.npz', record['outputs']['requests.npz']))
    if len(requests['identity']) == 0:
        raise ValueError('No native match requests in the authenticated identity inventory')
    index_keys = ('reference_row', 'obs') if args.minesweeper else ('sdss_id', 'distance_row', 'stellar_row')
    if args.minesweeper and (record['native_TYPE_unverified'] != len(requests['identity'])
                            or not (requests['obs'] == 'LCO').all()):
        raise ValueError('DR20 requests must match the LCO identity census')
    out = root/f'{prefix}_native'
    if out.exists():
        raise FileExistsError(out)
    native._IDS, native._OUT = requests['identity'], out
    if len(np.unique(native._IDS)) != len(native._IDS):
        raise ValueError('Duplicate requested Gaia IDs')
    bricks = native.SOUTH/'survey-bricks-dr11-south.fits.gz'
    b = uppercase_fields(fitsio.read(str(bricks), columns=['BRICKNAME', 'RA', 'DEC']))
    pos = unit_vectors(requests['ra'], requests['dec'])
    neighbors = cKDTree(unit_vectors(b['RA'], b['DEC'])).query_ball_point(
        pos, 2*np.sin(np.deg2rad(.4)/2))
    owners = {}
    for i, matched in enumerate(neighbors):
        for j in matched:
            owners.setdefault(j, []).append(i)
    tasks = [(name, np.asarray(owners[j])) for j, name in enumerate(np.char.strip(b['BRICKNAME'].astype('U')))
             if j in owners]
    (out/'bricks').mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('fork')) as workers:
        files = list(workers.map(native.brick, tasks))
    save_json(out/'native_sources.json', files)
    roles = ROOT/'manifests/v3_1/validation_roles/complete.json'
    prior = json.loads(roles.read_text())
    history = [verified(ROOT/name, prior['inputs'][name]) for name, _ in HISTORY]
    history += [verified(ROOT/'data/v3_1/validation_roles'/name, prior['outputs'][name])
                for name in ('field_photometry.npz', 'common_photometry.npz')]
    configs = [Path(p) for p in record['inputs'] if Path(p).parent == ROOT/'configs']
    bounds = list(NEW_SKY)
    bounds.extend(json.loads(verified(p, record['inputs'][str(p)]).read_text())['bounds_deg'] for p in configs)
    tables = [load(verified(out/'bricks'/f"{r['brick']}.npz", r['sha256']))
              for r in files if r.get('matched_primary')]
    counts = dict(requests=len(native._IDS), targeted_bricks=len(tasks),
                  missing_brick_files=sum(r.get('missing_file', False) for r in files),
                  missing_TYPE=sum(r.get('missing_TYPE', 0) for r in files),
                  exact_primary_rows=0, nonPSF=0, duplicate_primary=0, wrong_release=0,
                  failed_photometric_entry=0, historical_excluded=0, admitted=0)
    history_rows = [dict(path=str(p.relative_to(ROOT)), identity_overlap=0, sky_overlap=0) for p in history]
    outputs = ['native_sources.json']
    if tables:
        data = {key: np.concatenate([t[key] for t in tables]) for key in tables[0]}
        ids, ls_ids = data['identity'], data['ls_id']
        match = unique_join(ids, native._IDS)
        if (match < 0).any():
            raise ValueError('Native result outside the frozen identity requests')
        distinct = (unique_join(ids, ids) >= 0) & (unique_join(ls_ids, ls_ids) >= 0)
        native_psf = data['TYPE'] == 'PSF'
        release = data['RELEASE'] == 11010
        phot = data['photometric_entry']
        seen, history_rows = historical_exclusions(data, history)
        seen |= footprint(data['RA'], data['DEC'], bounds)
        allow = distinct & native_psf & release & phot & ~seen
        counts.update(exact_primary_rows=len(ids), nonPSF=int((~native_psf & (data['TYPE'] != '')).sum()),
                      duplicate_primary=int((~distinct).sum()), wrong_release=int((~release).sum()),
                      failed_photometric_entry=int((distinct & native_psf & release & ~phot).sum()),
                      historical_excluded=int((distinct & native_psf & release & phot & seen).sum()),
                      admitted=int(allow.sum()))
        accepted = {key: column[allow] for key, column in data.items()}
        if not (accepted['TYPE'] == 'PSF').all() or not (accepted['RELEASE'] == 11010).all():
            raise ValueError('Native sample export requires verified DR11 PSF')
        atomic_npz(out/'photometry.npz', **accepted)
        matched = match[allow]
        atomic_npz(out/'reference_index.npz', identity=ids[allow], ls_id=ls_ids[allow], TYPE=accepted['TYPE'],
                   **{key: requests[key][matched] for key in index_keys})
        outputs += ['photometry.npz', 'reference_index.npz']
    csv_rows(out/'history.csv', history_rows)
    save_json(out/'counts.json', counts)
    outputs += ['history.csv', 'counts.json']
    sources = history+configs+[bricks, roles, marker, source/'requests.npz', Path(__file__).resolve(),
                               ROOT/'scripts/v3_1_external_native.py', ROOT/'scripts/v3_0_native.py',
                               ROOT/'scripts/v3_2_reference_inventory.py']
    if args.minesweeper:
        sources.append(ROOT/'scripts/v3_2_minesweeper_inventory.py')
    save_json(out/'complete.json', dict(
        inputs={str(p): sha(p) for p in sources}, outputs={name: sha(out/name) for name in outputs},
        job=os.getenv('SLURM_JOB_ID'), observations_read=True, reference_values_read=False,
        all_admitted_PSF=True, validation_roles_frozen=False, **counts))


if __name__ == '__main__':
    main()

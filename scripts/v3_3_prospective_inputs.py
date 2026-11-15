#!/usr/bin/env python3
"""Register photometry-only validation objects before their outcomes are read."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_0_cohorts import INPUT_KEYS
from v3_0_native import NEW_SKY, native_payload, read_native
from v3_1_external_history import HISTORY, identity_position
from v3_2_reference_inventory import footprint
from v3_3_sed import verified

CONFIG = ROOT/'configs/v3_3_prospective.json'
OUT = ROOT/'data/v3_3/prospective/inputs'
SOUTH = Path('/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south/sweep/11.0')
ADDITIONAL_HISTORY = ('data/v3_1/validation_roles/field_photometry.npz',
                      'data/v3_1/validation_roles/common_photometry.npz',
                      'data/v3_2/minesweeper_native/photometry.npz',
                      'data/v3_2/common_scale_inventory/photometry.npz')


def history():
    manifests = ROOT/'manifests'
    v31 = json.loads((manifests/'v3_1/validation_roles/complete.json').read_text())
    compact = [verified(ROOT/p, v31['inputs'][p]) for p, _ in HISTORY]
    for path in ADDITIONAL_HISTORY:
        root = Path(path).parent
        receipt = json.loads((manifests / Path(*root.parts[1:]) / 'complete.json').read_text())
        compact.append(verified(ROOT/path, receipt['outputs'][Path(path).name]))
    identities, cells = [], []
    for path in compact:
        ids, sky = identity_position(path)
        identities.append(ids)
        cells.append(sky['cell'][sky['valid_coordinates']])
    return np.unique(np.concatenate(identities)), np.unique(np.concatenate(cells)), compact


def select(t, previously_seen, cells, outside_configs, groups, config):
    ids, lsids = t['identity'], t['ls_id']
    m = (t['TYPE'] == 'PSF') & t['photometric_entry'] & (t['RELEASE'] == 11010)
    m &= (ids > 0) & ~np.isin(ids, previously_seen) & ~np.isin(t['cell'], cells) & outside_configs
    duplicates, count = np.unique(ids[m], return_counts=True)
    m &= ~np.isin(ids, duplicates[count > 1])
    rows = np.flatnonzero(m)
    annotation = annotate_holdouts(groups, ids[rows], t['RA'][rows], t['DEC'][rows])
    rows = rows[~annotation['external_any']]
    inventory, selected, weights = [], [], []
    edges = config['observed_r_edges']
    for cell in np.unique(t['cell'][rows]):
        for lo, hi in zip(edges[:-1], edges[1:]):
            available = rows[(t['cell'][rows] == cell) & (t['r'][rows] >= lo) & (t['r'][rows] < hi)]
            ix = np.argsort(stable_rank(ids[available], config['seed']), kind='stable')
            take = available[ix[:config['maximum_per_field_magnitude_cell']]]
            inventory.append(dict(cell=int(cell), r_min=lo, r_max=hi, eligible=len(available), selected=len(take)))
            if len(take):
                selected.append(take)
                weights.append(np.full(len(take), len(available)/len(take)))
    if not selected:
        return {}, inventory
    rows = np.concatenate(selected)
    selected = {k: t[k][rows] for k in INPUT_KEYS}
    selected['observed_r'] = t['r'][rows]
    selected['sample_weight'] = np.concatenate(weights)
    if (lsids[rows] <= 0).any() or not (selected['TYPE'] == 'PSF').all():
        raise ValueError('Invalid native PSF or Legacy identity')
    return selected, inventory


def main():
    require_compute()
    config = json.loads(CONFIG.read_text())
    previous, old_cells, sources = history()
    bounds, configs = list(NEW_SKY), []
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if 'bounds_deg' in cfg:
            bounds.append(cfg['bounds_deg'])
            configs.append(path)
    group_dir = ROOT/'data/reference'
    group_paths = [group_dir/name for name in ('holdout_references_v1.npz', 'holdout_group_names.json')]
    groups = load_reference_catalogs(group_dir)
    chunks, census, sweeps = [], [], {}
    for task in config['sweeps']:
        path = SOUTH/task['file']
        if not path.exists():
            census.append(dict(sweep=path.name, missing_sweep=True, selected=0))
            continue
        raw = read_native(path)
        lo, de, hi, dh = task['bounds_deg']
        windows = ((raw['RA'] >= lo) & (raw['RA'] < hi) & (raw['DEC'] >= de) & (raw['DEC'] < dh))
        t = native_payload(raw[windows], 'DECam_DR11')
        del raw
        selection, strata = select(t, previous, old_cells, ~footprint(t['RA'], t['DEC'], bounds), groups, config)
        if selection:
            chunks.append(selection)
        census.append(dict(sweep=path.name, parent=len(t['TYPE']), non_PSF=int(((t['TYPE'] != 'PSF') & (t['TYPE'] != '')).sum()),
                           missing_TYPE=int((t['TYPE'] == '').sum()), selected=sum(s['selected'] for s in strata), strata=strata))
        stat = path.stat()
        sweeps[str(path)] = [stat.st_size, stat.st_mtime_ns]
    if not chunks:
        raise ValueError(f'No unused PSFs pass the registered selection: {census}')
    data = {key: np.concatenate([p[key] for p in chunks]) for key in chunks[0]}
    for key in ('identity', 'ls_id'):
        if len(np.unique(data[key])) != len(data[key]):
            raise ValueError(f'Duplicate validation {key}')
    OUT.mkdir(parents=True, exist_ok=False)
    photo = OUT/'photometry.npz'
    atomic_npz(photo, **data)
    save_json(OUT/'complete.json', dict(n=len(data['identity']), all_PSF=True, census=census,
        sources={str(p.relative_to(ROOT)): sha(p) for p in [CONFIG, Path(__file__).resolve(), *configs, *sources, *group_paths]},
        native_sweeps=sweeps, known_role_overlap=0, known_cell_overlap=0, sha256=sha(photo),
        outcomes_read=False, job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()

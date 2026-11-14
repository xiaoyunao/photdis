#!/usr/bin/env python3
"""Inventory DR20 BOSS MINESweeper identities with reference values closed."""
import json
import os
from pathlib import Path

import fitsio
import numpy as np

from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3_1.sky import qualified_sky
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_0_native import NEW_SKY
from v3_1_external_history import HISTORY, identity_position
from v3_2_reference_inventory import footprint, text
from v3_2_tilt_inputs import verified

COLUMNS = ('source_id', 'ra', 'dec', 'obs', 'mjd', 'n_exp')
FILENAME = 'minesweeper_v1.2.2.fits'
DIGEST = '65bab7406656d17d335beb69ee56bcd7499cb2130fc96b783f27ac7bb0a30b15'


def qualify_field_requests(ids, ra, dec, highlat, v1, old_products):
    sky = qualified_sky(ra, dec)
    history = [verified(ROOT/name, v1['inputs'][name]) for name, _ in HISTORY] + old_products
    typed_path = ROOT/'data/giant_parameter_audit_v1/typed_candidates.json'
    audit = json.loads((ROOT/'manifests/v3_1/external_history/complete.json').read_text())
    verified(typed_path, audit['inputs'][str(typed_path.relative_to(ROOT))])
    typed = json.loads(typed_path.read_text())
    occupied = set()
    exposed = np.isin(ids, np.array([int(row['source_id']) for row in typed['rows']], dtype='i8'))
    rows = []
    for path in history:
        source, old_sky = identity_position(path)
        blocks = np.unique(old_sky['cell'][old_sky['valid_coordinates']])
        occupied.update(blocks.tolist())
        direct = np.isin(ids, source[source > 0])
        exposed |= direct
        rows.append(dict(path=str(path.relative_to(ROOT)), matched_identity=int(direct.sum()),
                         matched_sky_block=int(np.isin(sky['cell'], blocks).sum())))
    good = highlat & ~exposed & ~np.isin(sky['cell'], list(occupied))
    configs = []
    bounds = list(NEW_SKY)
    for path in sorted((ROOT/'configs').glob('*.json')):
        values = json.loads(path.read_text())
        if 'bounds_deg' in values:
            configs.append(path)
            bounds.append(values['bounds_deg'])
    field = footprint(ra, dec, bounds)
    good &= ~field
    group_root = ROOT/'data/reference'
    group_paths = [group_root/name for name in ('holdout_references_v1.npz', 'holdout_group_names.json')]
    for path in group_paths:
        verified(path, audit['inputs'][str(path.relative_to(ROOT))])
    group = annotate_holdouts(load_reference_catalogs(group_root), ids[good], ra[good], dec[good])
    field_rows = np.flatnonzero(good)[~group['external_any']]
    counts = dict(high_latitude=int(highlat.sum()), prior_id_excluded=int((highlat & exposed).sum()),
                  old_sky_excluded=int((highlat & ~exposed & np.isin(sky['cell'], list(occupied))).sum()),
                  footprint_excluded=int((highlat & ~exposed & ~np.isin(sky['cell'], list(occupied)) & field).sum()),
                  known_system_excluded=int(group['external_any'].sum()),
                  match_requests=len(field_rows), scientific_admissions=0)
    return field_rows, sky['cell'], counts, rows, history + group_paths + configs + [typed_path]


def main():
    require_compute()
    root = ROOT/'data/v3_2'
    path = verified(root/'reference_sources/dr20_minesweeper'/FILENAME, DIGEST)
    with fitsio.FITS(str(path)) as fits:
        names = {name.lower(): name for name in fits[1].get_colnames()}
        metadata = fits[1].read(columns=[names[k] for k in COLUMNS])
    ids, ra, dec, obs, mjd, visits = (metadata[names[k]] for k in COLUMNS)
    if ids.dtype.kind not in 'iu':
        raise ValueError('Exact Gaia DR3 integer identities required')
    sky = qualified_sky(ra, dec)
    lco = (text(obs) == 'LCO') & (visits > 0) & (mjd > 0) & (mjd <= 60708)
    eligible = np.flatnonzero((ids > 0) & lco & sky['valid_coordinates'])
    distinct = unique_join(ids[eligible], ids[eligible]) >= 0
    unique = eligible[distinct]
    role_path = ROOT/'manifests/v3_1/validation_roles/complete.json'
    roles = json.loads(role_path.read_text())
    old = [verified(ROOT/'data/v3_1/validation_roles'/name, roles['outputs'][name])
           for name in ('field_photometry.npz', 'common_photometry.npz')]
    field, cells, counts, history, sources = qualify_field_requests(
        ids[unique], ra[unique], dec[unique], abs(sky['latitude'][unique]) > 30, roles, old)
    now = unique[field]
    out = root/'minesweeper_inventory'
    out.mkdir(exist_ok=False)
    atomic_npz(out/'requests.npz', identity=ids[now], ra=ra[now], dec=dec[now],
               cell=cells[field], reference_row=now, obs=text(obs[now]))
    csv_rows(out/'history.csv', history)
    counts.update(archive_rows=len(ids), eligible_lco=len(eligible),
                  duplicate_lco_rows=int((~distinct).sum()), unique_lco=len(unique))
    save_json(out/'counts.json', counts)
    sources += [path, role_path, Path(__file__).resolve(),
                ROOT/'docs/v3_2_minesweeper_intake.md',
                ROOT/'scripts/v3_2_reference_inventory.py',
                ROOT/'scripts/v3_1_external_history.py',
                ROOT/'src/photdis/v3_1/sky.py', ROOT/'src/photdis/holdouts.py',
                ROOT/'manifests/v3_1/external_history/complete.json']
    save_json(out/'complete.json', dict(
        inputs={str(p): sha(p) for p in sources}, outputs={p.name: sha(p) for p in out.iterdir()},
        metadata_columns=list(COLUMNS), job=os.getenv('SLURM_JOB_ID'),
        native_TYPE_unverified=len(now), outcomes_read=False, scientific_sample_exported=False))


if __name__ == '__main__':
    main()

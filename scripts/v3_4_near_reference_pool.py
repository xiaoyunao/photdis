#!/usr/bin/env python3
"""Persist exact unopened Gaia candidates and an additional DR20 metadata match."""
import json
import os
from pathlib import Path

import fitsio
import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_2_minesweeper_inventory import COLUMNS, DIGEST, FILENAME
from v3_2_reference_inventory import text
from v3_3_sed import verified
from v3_4_near_development_inputs import CONFIG, OUT as CENSUS
from v3_4_near_reference_support import OUT as PREVIOUS, native_field, past_evidence, separation

OUT = CENSUS.parent/'reference_pool'


def match_apo(frame, unique_gaia, history_exposed, path):
    with fitsio.FITS(str(path)) as f:
        names = {k.lower(): k for k in f[1].get_colnames()}
        rows = f[1].read(columns=[names[k] for k in COLUMNS])
    sid, ra, dec, obs, mjd, visits = (rows[names[k]] for k in COLUMNS)
    if sid.dtype.kind not in 'iu':
        raise ValueError('Expected exact integer DR3 source IDs')
    eligible = np.flatnonzero((sid > 0) & (text(obs) == 'APO') & (mjd > 0) & (mjd <= 60708) & (visits > 0)
                              & np.isfinite(ra) & np.isfinite(dec) & (abs(dec) <= 90))
    match = unique_join(frame['source_id'], sid[eligible])
    found = np.flatnonzero(unique_gaia & (match >= 0))
    ind = eligible[match[found]]
    close = separation(frame['RA'][found], frame['DEC'][found], ra[ind], dec[ind])
    found, ind = found[close], ind[close]
    return dict({k: v[found] for k, v in frame.items()}, reference_row=ind,
                history_exposed=history_exposed[found], observatory=np.full(len(found), 'APO'))


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    marker = CENSUS/'complete.json'
    census = json.loads(marker.read_text())
    inventory_marker = PREVIOUS/'complete.json'
    past_inventory = json.loads(inventory_marker.read_text())
    inputs = {str(p): sha(p) for p in (CONFIG, marker, inventory_marker, Path(__file__).resolve())}
    ids, ls_ids = past_evidence(inputs)
    fields = [native_field(field, cfg, census, inputs) for field in cfg['fields']]
    frame = {k: np.concatenate([f[k] for f in fields]) for k in fields[0]}
    if not (frame['TYPE'] == 'PSF').all() or len(np.unique(frame['identity'])) != census['parent']:
        raise ValueError('Native application frame disagrees with its inventory')
    sid = frame['source_id']
    unique_gaia = (sid > 0) & (unique_join(sid, sid) >= 0)
    overlap = np.isin(sid, ids[ids > 0]) | np.isin(frame['identity'], ids) | np.isin(frame['ls_id'], ls_ids)
    reserved = unique_gaia & ~overlap
    if int(reserved.sum()) != past_inventory['gaia_identity_outside_prior_evidence']:
        raise ValueError('Unopened Gaia count disagrees with independently run metadata inventory')
    p = verified(CENSUS/'photometry.npz', census['outputs']['photometry.npz'])
    inputs[str(p)] = census['outputs']['photometry.npz']
    photo = load(p)
    sample = photo['application_sample']
    ix = unique_join(photo['identity'][sample], frame['identity'])
    if (ix < 0).any():
        raise ValueError('Some probability-sampled application sources are missing')
    for k in ('ls_id', 'TYPE', 'stratum', 'field'):
        np.testing.assert_array_equal(photo[k][sample], frame[k][ix])
    design = dict({k: v[ix] for k, v in frame.items()}, history_exposed=overlap[ix],
                  gaia_unique=unique_gaia[ix], design_weight=photo['application_weight'][sample])
    path = verified(ROOT/'data/v3_2/reference_sources/dr20_minesweeper'/FILENAME, DIGEST)
    inputs[str(path)] = DIGEST
    apo = match_apo(frame, unique_gaia, overlap, path)
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_npz(OUT/'gaia_pool.npz', **{k: v[reserved] for k,v in frame.items()})
    atomic_npz(OUT/'application_gaia_design.npz', **design)
    atomic_npz(OUT/'DR20_APO.npz', **apo)
    csv_rows(OUT/'coverage.csv', [dict(field=i, gaia_pool=int((reserved & (frame['field'] == i)).sum()),
                representative_sample=int((design['field'] == i).sum()),
                sample_gaia_links=int(((design['field'] == i) & design['gaia_unique']).sum()),
                sample_gaia_unopened=int(((design['field'] == i) & design['gaia_unique'] & ~design['history_exposed']).sum()),
                APO_matches=int((apo['field'] == i).sum()),
                APO_outside_history=int(((apo['field'] == i) & ~apo['history_exposed']).sum())) for i in cfg['fields']])
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), new_outcomes_read=False, all_PSF=True,
        inputs=inputs, outputs={p.name:sha(p) for p in sorted(OUT.iterdir())}, gaia_pool=int(reserved.sum()),
        representative_sample=len(ix), representative_gaia_unopened=int((design['gaia_unique'] & ~design['history_exposed']).sum()),
        APO_matches=len(apo['identity']), APO_outside_history=int((~apo['history_exposed']).sum()),
        interpretation='Metadata only. Gaia reserve is a subset outside historical evidence, and spectral inclusion remains targeted.'))


if __name__ == '__main__':
    main()

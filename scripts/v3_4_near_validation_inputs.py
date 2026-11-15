#!/usr/bin/env python3
"""Build paired application, fresh Gaia and targeted-spectrum validation rosters."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_development_inputs import OUT as CENSUS
from v3_4_near_reference_pool import OUT as POOL
from v3_4_near_reference_support import OUT as SUPPORT

CONFIG = ROOT/'configs/v3_4_near_validation.json'
OUT = CENSUS.parent/'validation'/'inputs'


def receipt_payload(folder, name, inputs):
    marker = folder/'complete.json'
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    r = json.loads(marker.read_text())
    if not r['all_PSF'] or r['new_outcomes_read']:
        raise ValueError('Roster parent must contain only closed native PSF evidence')
    p = verified(folder/name, r['outputs'][name])
    inputs[str(p.relative_to(ROOT))] = r['outputs'][name]
    return load(p)


def fresh_draw(pool, cfg):
    field_cell = np.column_stack((pool['field'], pool['stratum']))
    groups, ids = np.unique(field_cell, axis=0, return_inverse=True)
    ids = ids.ravel()
    rank = stable_rank(pool['source_id'], cfg['fresh_gaia_sample_salt'])
    order = np.lexsort((rank, ids))
    blocks = np.split(order, np.flatnonzero(np.diff(ids[order]))+1)
    picked, weight, inventory = [], [], []
    for (field, cell), block in zip(groups, blocks):
        q = block[:cfg['fresh_gaia_cap_per_field_cell']]
        picked.append(q)
        weight.append(np.full(len(q), len(block)/len(q)))
        inventory.append(dict(field=int(field), stratum=int(cell), pool_n=len(block), sampled=len(q)))
    return np.concatenate(picked), np.concatenate(weight), inventory


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (CONFIG, Path(__file__))}
    application = receipt_payload(POOL, 'application_gaia_design.npz', inputs)
    pool = receipt_payload(POOL, 'gaia_pool.npz', inputs)
    spectra = {name: receipt_payload(folder, name+'.npz', inputs) for folder, name in (
        (SUPPORT, 'DR20_MINESweeper'), (SUPPORT, 'DR19_Astra'), (POOL, 'DR20_APO'))}
    picks, weights, inventory = fresh_draw(pool, cfg)
    fresh = {k: v[picks] for k, v in pool.items()}
    fresh['design_weight'] = weights
    np.testing.assert_allclose(weights.sum(), len(pool['identity']), rtol=0, atol=1e-7)

    requested = np.unique(np.concatenate([p['identity'] for p in (application, fresh, *spectra.values())]))
    census = json.loads((CENSUS/'complete.json').read_text())
    parts = []
    for field in sorted(np.unique(np.concatenate([p['field'] for p in (application, fresh, *spectra.values())]))):
        key = f'data/v3_0/catalog/field_{field:02d}/inputs.npz'
        photo = load(verified(ROOT/key, census['inputs'][key]))
        inputs[key] = census['inputs'][key]
        use = np.isin(photo['identity'], requested)
        parts.append({k: v[use] for k, v in photo.items()})
    combined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(combined['identity'])
    combined = {k: v[order] for k, v in combined.items()}
    np.testing.assert_array_equal(combined['identity'], requested)
    if not (combined['TYPE'] == 'PSF').all():
        raise ValueError('Expected unique verified Legacy PSFs')
    for p in (application, fresh, *spectra.values()):
        ix = np.searchsorted(combined['identity'], p['identity'])
        np.testing.assert_array_equal(p['ls_id'], combined['ls_id'][ix])
        p['prediction_row'] = ix

    OUT.mkdir(parents=True, exist_ok=False)
    atomic_npz(OUT/'photometry.npz', **combined)
    for name, data in [('application', application), ('fresh_gaia', fresh), *spectra.items()]:
        atomic_npz(OUT/f'{name}.npz', **data)
    csv_rows(OUT/'fresh_gaia_design.csv', inventory)
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), all_PSF=True,
        new_outcomes_read=False, n=len(requested), application_n=len(application['identity']),
        fresh_gaia_n=len(fresh['identity']), fresh_gaia_pool=len(pool['identity']),
        spectrum_n={name:len(p['identity']) for name,p in spectra.items()},
        inputs=inputs, outputs={p.name:sha(p) for p in sorted(OUT.iterdir())}))


if __name__ == '__main__':
    main()

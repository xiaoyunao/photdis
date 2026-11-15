#!/usr/bin/env python3
"""Identify archived spectroscopic opportunities in the application denominator."""
import json
import os
from pathlib import Path

import fitsio
import numpy as np

from photdis.assembly_v1_4 import unit_vectors
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_2_minesweeper_inventory import COLUMNS, DIGEST, FILENAME
from v3_2_reference_inventory import ARCHIVES, inventory_join, read_apogee, text
from v3_3_prospective_inputs import history
from v3_3_sed import BASE, MANIFESTS, verified
from v3_4_near_development_inputs import CONFIG, OUT as CENSUS, cells

OUT = CENSUS.parent/'reference_support'


def spectroscopic_inventory(inputs):
    archive_root = ROOT/'data/v3_2/reference_sources'
    path = verified(archive_root/'dr20_minesweeper'/FILENAME, DIGEST)
    inputs[str(path)] = DIGEST
    with fitsio.FITS(str(path)) as fits:
        names = {name.lower(): name for name in fits[1].get_colnames()}
        metadata = fits[1].read(columns=[names[k] for k in COLUMNS])
    ids, ra, dec, obs, mjd, visits = (metadata[names[k]] for k in COLUMNS)
    if ids.dtype.kind not in 'iu':
        raise ValueError('Expected exact integer Gaia IDs')
    k = np.flatnonzero((ids > 0) & (text(obs) == 'LCO') & (visits > 0) & (mjd > 0) & (mjd <= 60708)
                      & np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90))
    k = k[unique_join(ids[k], ids[k]) >= 0]
    mine = dict(source_id=ids[k], RA=ra[k], DEC=dec[k], distance_row=k, stellar_row=k)

    receipt = ROOT/'manifests/v3_2/reference_sources/dr19_acquisition.json'
    atlases = json.loads(receipt.read_text())['archives']
    inputs[str(receipt)] = sha(receipt)
    bank = {}
    for kind, name in ARCHIVES.items():
        p = verified(archive_root/'dr19_astra_0.6.0'/name, atlases[name]['sha256'])
        inputs[str(p)] = atlases[name]['sha256']
        bank[kind] = read_apogee(p)
    old = ROOT/'manifests/v3_1/external_inventory_finite_sky/complete.json'
    inputs[str(old)] = sha(old)
    legacy = json.loads(old.read_text())['inputs']
    old_name, = [n for n in legacy if Path(n).name == 'allStar-dr17-synspec_rev1.fits']
    path = verified(Path(old_name), legacy[old_name])
    inputs[str(path)] = legacy[old_name]
    dr17 = fitsio.read(str(path), ext=1, columns=['APOGEE_ID', 'GAIAEDR3_SOURCE_ID'])
    a, b, _ = inventory_join(bank['distance'], bank['stellar'],
        dr17['GAIAEDR3_SOURCE_ID'][dr17['GAIAEDR3_SOURCE_ID'] > 0],
        np.unique(text(dr17['APOGEE_ID'])[text(dr17['APOGEE_ID']) != '']))
    d, spec = bank['distance'], bank['stellar']
    sky_ra, sky_dec = d['ra'][a], d['dec'][a]
    good = np.isfinite(sky_ra) & np.isfinite(sky_dec) & np.isfinite(spec['ra'][b]) & np.isfinite(spec['dec'][b])
    good &= (np.abs(sky_dec) <= 90) & (np.abs(spec['dec'][b]) <= 90)
    a, b = a[good], b[good]
    close = separation(d['ra'][a], d['dec'][a], spec['ra'][b], spec['dec'][b])
    a, b = a[close], b[close]
    astra = dict(source_id=d['gaia_dr3_source_id'][a], RA=d['ra'][a], DEC=d['dec'][a],
                distance_row=a, stellar_row=b, sdss_id=d['sdss_id'][a])
    return {'DR20_MINESweeper': mine, 'DR19_Astra': astra}


def separation(ra, dec, target_ra, target_dec):
    return np.linalg.norm(unit_vectors(ra, dec)-unit_vectors(target_ra, target_dec), axis=1) <= 2*np.sin(np.deg2rad(1/7200))


def native_field(field, cfg, census, inputs):
    key = f'data/v3_0/catalog/field_{field:02d}/inputs.npz'
    digest = census['inputs'][key]
    inputs[key] = digest
    photo = load(verified(ROOT/key, digest))
    marker = MANIFESTS/'application_inputs'/f'field_{field:02d}.json'
    digest = json.loads(marker.read_text())['sha256']
    parent = marker.name.replace('.json', '.npz')
    path = verified(BASE/'application_inputs'/parent, digest)
    inputs[str(marker)] = sha(marker)
    inputs[str(path)] = digest
    with np.load(path) as arrays:
        rows = unique_join(photo['parent_row'], arrays['parent_row'])
        if (rows < 0).any():
            raise ValueError('Missing unique DR11 native source row')
        for k in ('ls_id', 'TYPE', 'identity'):
            np.testing.assert_array_equal(photo[k], arrays[k][rows])
        native = arrays['source_id'][rows]
    group, r, gr = cells(photo, cfg)
    return dict({k: photo[k] for k in ('identity', 'ls_id', 'TYPE', 'RA', 'DEC', 'field')},
                source_id=native, stratum=group, observed_r=r, dereddened_g_r=gr)


def past_evidence(inputs):
    past, _, paths = history()
    for path in paths:
        inputs[str(path)] = sha(path)
    manifests = ROOT/'manifests'
    co = json.loads((MANIFESTS/'cohorts.json').read_text())
    v33 = json.loads((manifests/'v3_3/prospective_inputs.json').read_text())
    historical = [verified(BASE/'cohorts/application.npz', co['outputs']['cohorts/application.npz']),
                  verified(ROOT/'data/v3_3/prospective/inputs/photometry.npz', v33['sha256'])]
    all_ids = [past]
    all_legacy = []
    for path in [*paths, *historical]:
        if path.suffix != '.npz':
            continue
        with np.load(path) as t:
            if 'ls_id' in t:
                all_legacy.append(t['ls_id'])
            if 'source_id' in t:
                all_ids.append(t['source_id'])
            if 'identity' in t:
                all_ids.append(t['identity'])
    for path in historical:
        inputs[str(path)] = sha(path)
    return np.unique(np.concatenate(all_ids)), np.unique(np.concatenate(all_legacy))


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    marker = CENSUS/'complete.json'
    census = json.loads(marker.read_text())
    inputs = {str(p): sha(p) for p in (CONFIG, marker, Path(__file__))}
    spectra = spectroscopic_inventory(inputs)
    previous_id, previous_legacy = past_evidence(inputs)
    fields = [native_field(f, cfg, census, inputs) for f in cfg['fields']]
    frame = {k: np.concatenate([t[k] for t in fields]) for k in fields[0]}
    if len(np.unique(frame['ls_id'])) != len(frame['identity']) or len(frame['ls_id']) != census['parent']:
        raise ValueError('Native denominator count or identity failed')
    identity = frame['source_id']
    direct = (identity > 0) & (unique_join(identity, identity) >= 0)
    known = (np.isin(frame['identity'], previous_id) | np.isin(identity, previous_id[previous_id > 0])
             | np.isin(frame['ls_id'], previous_legacy))
    masks, outputs = {}, {}
    for name, source in spectra.items():
        ix = unique_join(identity, source['source_id'])
        rows = np.flatnonzero(direct & (ix >= 0))
        j = ix[rows]
        close = separation(frame['RA'][rows], frame['DEC'][rows], source['RA'][j], source['DEC'][j])
        rows, j = rows[close], j[close]
        matched = np.zeros(len(identity), bool)
        matched[rows] = True
        masks[name] = matched
        available = {k: v[rows] for k, v in frame.items()}
        available.update({k: source[k][j] for k in source if k not in ('RA', 'DEC', 'source_id')})
        available['historical_evidence'] = known[rows]
        outputs[name] = available
    blocks = []
    for field in cfg['fields']:
        f = frame['field'] == field
        for group in np.unique(frame['stratum'][f]):
            mask = f & (frame['stratum'] == group)
            record = dict(field=field, stratum=int(group), parent=int(mask.sum()),
                          unique_native_gaia=int((mask & direct).sum()),
                          gaia_identity_outside_prior_evidence=int((mask & direct & ~known).sum()))
            for name, v in masks.items():
                record[f'{name}_matched'] = int((mask & v).sum())
                record[f'{name}_outside_prior_evidence'] = int((mask & v & ~known).sum())
            blocks.append(record)
    OUT.mkdir(parents=True, exist_ok=False)
    for name, data in outputs.items():
        atomic_npz(OUT/f'{name}.npz', **data)
    csv_rows(OUT/'coverage.csv', blocks)
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), parent=len(identity),
        all_PSF=True, new_outcomes_read=False, inputs=inputs, outputs={p.name: sha(p) for p in sorted(OUT.iterdir())},
        unique_native_gaia=int(direct.sum()), gaia_identity_outside_prior_evidence=int((direct & ~known).sum()),
        matches={name:dict(n=int(v.sum()), outside_prior_evidence=int((v & ~known).sum())) for name,v in masks.items()},
        interpretation='Targeted spectrum metadata intersections. Final qualification and outcomes remain closed. '
                       'Reference targeting is not a probability sample.'))


if __name__ == '__main__':
    main()

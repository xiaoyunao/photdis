#!/usr/bin/env python3
"""Run the registered independent finite-distance extinction validation."""
import argparse
import json
from itertools import combinations

from astropy.coordinates import SkyCoord
import numpy as np

import environment_apogee as apogee
import environment_dust as dust
import environment_pilot as pilot
import environment_results as results
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_4_near_forecast import prepare

CONFIG = ROOT/'configs/environment_transfer_20261120.json'
REGISTRY = json.loads(CONFIG.read_text())
OLD = ROOT/'data/environment_20261120'
OUT = ROOT/'data/environment_transfer_20261120'
TABLES = OUT/'results'
C = dict(pilot.C, fields=REGISTRY['fields'])
FROZEN = OUT/'preflight/freeze.json'


def configure():
    pilot.CONFIG = CONFIG
    for module in (pilot, results, dust, apogee):
        module.C = C
        module.OUT = OUT


def preflight():
    if FROZEN.exists():
        raise FileExistsError(FROZEN)
    history = json.loads((ROOT/'configs/environment_20261120.json').read_text())
    fields = C['fields'] + history['fields']
    for a, b in combinations(fields, 2):
        ca, cb = [SkyCoord(*f[1:], frame='galactic', unit='deg') for f in (a, b)]
        assert ca.separation(cb).deg > 2*C['field_radius_deg'], (a, b)
    t = load(OUT/'census/qualified.npz')
    old = load(OLD/'forecast/roster.npz')
    models, _, sources = prepare()
    model = models['qrf']
    trained = np.isin(t['identity'], model.training_identity) | np.isin(t['ls_id'], -model.training_identity)
    duplicate = np.isin(t['ls_id'], old['ls_id'])
    duplicate |= (t['source_id'] > 0) & np.isin(t['source_id'], old['source_id'][old['source_id'] > 0])
    clean = ~(trained | t['historical_overlap'] | duplicate)
    mag = np.pad(t['mag_grz'], ((0, 0), (0, 2)))
    supported = np.concatenate([pilot.support(model, a) for a in np.array_split(mag, 40)])
    strong = np.isfinite(t['gaia_distance_kpc'])
    areas, area_hash = results.area()
    blocks = results.covariance_blocks(t)
    rows, selected, dust_ids = [], [], {}
    for area in areas:
        name = area['field']
        field = t['field'] == name
        eligible = field & clean & strong & supported
        k = pilot.sample(t['ls_id'], eligible, C['reference_per_field_per_provider'], 10)
        j = pilot.sample(t['ls_id'], np.isin(t['ls_id'], t['ls_id'][k]), C['bayestar_iteration_per_field'], 20)
        dust_ids[name] = t['ls_id'][j].tolist()
        qualifies = len(j) >= REGISTRY['minimum_gaia'] and len(np.unique(blocks[j])) >= REGISTRY['minimum_blocks']
        role = REGISTRY['roles'][name]
        if qualifies and role != 'reserve':
            selected.append(name)
        rows.append(dict(**area, role=role, qualified_psf=int(field.sum()),
                         finite_gaia=int((field & strong).sum()),
                         qrf_support=int((field & supported).sum()),
                         training=int((field & trained).sum()),
                         historical=int((field & t['historical_overlap']).sum()),
                         old_field_overlap=int((field & duplicate).sum()),
                         clean_supported_gaia=int(eligible.sum()), dust_n=len(j),
                         blocks=len(np.unique(blocks[j])), qualifies=bool(qualifies),
                         median_ebv=float(np.median(t['EBV'][field])),
                         median_fracflux_r=float(np.median(t['FRACFLUX_R'][field])),
                         median_r=float(np.median(t['r'][field]))))
    if sum(REGISTRY['roles'][name] == 'low' for name in selected) < 2:
        selected += [r['field'] for r in rows if r['role'] == 'reserve' and r['qualifies']]
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    path = FROZEN.parent/'availability.csv'
    csv_rows(path, rows)
    atomic_npz(FROZEN.parent/'identity_masks.npz', ls_id=t['ls_id'], clean=clean, support=supported)
    save_json(FROZEN, dict(selected=selected, availability_sha256=sha(path),
                registry_sha256=sha(CONFIG), parent_config_sha256=sha(ROOT/'configs/environment_20261120.json'),
                census_sha256=sha(OUT/'census/complete.json'), area_sha256=area_hash,
                qrf_inputs={str(p):sha(p) for p in sources},
                identity_sha256=sha(FROZEN.parent/'identity_masks.npz'),
                dust_ids={name:dust_ids[name] for name in selected}))
    print('Pre-outcome freeze', sha(FROZEN), [(r['field'],r['dust_n'],r['blocks']) for r in rows], flush=True)


def operational_stage(stage):
    frozen = json.loads(FROZEN.read_text())
    assert frozen['registry_sha256'] == sha(CONFIG)
    assert frozen['census_sha256'] == sha(OUT/'census/complete.json')
    assert all(sha(p) == h for p, h in frozen['qrf_inputs'].items())
    C['fields'] = [f for f in REGISTRY['fields'] if f[0] in frozen['selected']]
    if stage == 'forecast':
        identity = FROZEN.parent/'identity_masks.npz'
        assert sha(identity) == frozen['identity_sha256']
        masks = load(identity)
        original = pilot.sample

        def sample(ids, selected, maximum, salt):
            if salt in (10, 11, 12):
                np.testing.assert_array_equal(ids, masks['ls_id'])
                selected = selected & masks['clean']
            return original(ids, selected, maximum, salt)

        pilot.sample = sample
        pilot.forecast()
    if stage == 'apogee':
        apogee.main()
    if stage == 'dust':
        t = load(OUT/'forecast/roster.npz')
        for name in frozen['selected']:
            j = pilot.sample(t['ls_id'], (t['field']==name)&t['gaia_evaluation'],
                             C['bayestar_iteration_per_field'], 20)
            np.testing.assert_array_equal(t['ls_id'][j], frozen['dust_ids'][name])
        dust.main()


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('census','preflight','forecast','dust','apogee'))
    stage = parser.parse_args().stage
    configure()
    if stage == 'census':
        pilot.census()
    elif stage == 'preflight':
        preflight()
    else:
        operational_stage(stage)


if __name__ == '__main__':
    main()

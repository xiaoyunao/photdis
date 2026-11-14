#!/usr/bin/env python3
"""Census of new DR19 APOGEE identities without reading reference outcomes."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import fitsio
import numpy as np

from photdis.assembly_v1_4 import unit_vectors
from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.reference_v1_4 import pair_keys, unique_join
from photdis.stage1 import require_compute
from photdis.v3_1.sky import qualified_sky
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_0_native import NEW_SKY
from v3_1_external_history import HISTORY, identity_position
from v3_2_tilt_inputs import verified


COLS = ('sdss_id', 'gaia_dr3_source_id', 'ra', 'dec', 'release', 'apred',
        'sdss4_apogee_id', 'obj', 'n_good_visits', 'telescope')
ARCHIVES = {'distance': 'astraAllStarAstroNNdist-0.6.0.fits.gz',
            'stellar': 'astraAllStarASPCAP-0.6.0.fits.gz'}
SIZE = {'distance': 534769970, 'stellar': 1171102556}


def text(column):
    return np.char.strip(np.asarray(column).astype('U'))


def read_apogee(path):
    """Read only the documented APOGEE HDU identity and provenance columns."""
    with fitsio.FITS(str(path)) as hdu:
        header = hdu[0].read_header()
        if str(header.get('V_ASTRA', '')).strip() != '0.6.0':
            raise ValueError(f'Unexpected Astra version in {path}')
        names = {name.lower(): name for name in hdu[2].get_colnames()}
        table = hdu[2].read(columns=[names[name] for name in COLS])
    return {key: table[names[key]] for key in COLS}


def inventory_join(distance, stellar, legacy_ids, legacy_names):
    """Reject duplicated positive identifiers and legacy lineage in both tables."""
    def eligible(table):
        sid, gid = table['sdss_id'], table['gaia_dr3_source_id']
        if sid.dtype.kind not in 'iu' or gid.dtype.kind not in 'iu':
            raise ValueError('SDSS and Gaia identities must remain exact integers')
        new = ((text(table['release']) == 'dr19') & (text(table['apred']) == '1.3')
               & (text(table['telescope']) == 'apo25m') & (table['n_good_visits'] > 0))
        legacy = np.isin(gid, legacy_ids) | np.isin(text(table['obj']), legacy_names)
        legacy |= np.isin(text(table['sdss4_apogee_id']), legacy_names)
        return (sid > 0) & (gid > 0) & new & ~legacy

    a, b = [np.flatnonzero(eligible(table)) for table in (distance, stellar)]
    def keys(table, rows):
        sid, gid = table['sdss_id'][rows], table['gaia_dr3_source_id'][rows]
        return pair_keys(sid, gid)

    eligible_counts = len(a), len(b)
    unique_rows = []
    for table, rows in [(distance, a), (stellar, b)]:
        consistent = np.ones(len(rows), dtype=bool)
        for name in ('sdss_id', 'gaia_dr3_source_id'):
            ids = table[name][rows]
            consistent &= unique_join(ids, ids) >= 0
        unique_rows.append(rows[consistent])
    a, b = unique_rows
    key_a, key_b = keys(distance, a), keys(stellar, b)
    j = unique_join(key_a, key_b)
    matched = j >= 0
    return a[matched], b[j[matched]], dict(
        distance_fresh_eligible=eligible_counts[0], stellar_fresh_eligible=eligible_counts[1],
        unmatched_stellar=int((~matched).sum()),
        distance_duplicate_identity_exclusions=eligible_counts[0]-len(a),
        stellar_duplicate_identity_exclusions=eligible_counts[1]-len(b))


def footprint(ra, dec, bounds):
    inside = np.zeros(len(ra), bool)
    for lo, de, hi, dh in bounds:
        in_ra = (ra >= lo) & (ra <= hi) if lo <= hi else (ra >= lo) | (ra <= hi)
        inside |= in_ra & (dec >= de) & (dec <= dh)
    return inside


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('distance_sha256')
    parser.add_argument('stellar_sha256')
    args = parser.parse_args()
    root = ROOT/'data/v3_2/reference_sources/dr19_astra_0.6.0'
    paths = {k: verified(root/name, getattr(args, f'{k}_sha256')) for k, name in ARCHIVES.items()}
    for key, path in paths.items():
        if path.stat().st_size != SIZE[key]:
            raise ValueError('Source archive size changed')
    r1 = ROOT/'manifests/v3_1/validation_roles/complete.json'
    v1 = json.loads(r1.read_text())
    role_root = ROOT/'data/v3_1/validation_roles'
    old_products = [verified(role_root/name, v1['outputs'][name])
                    for name in ('field_photometry.npz', 'common_photometry.npz')]
    legacy_record = ROOT/'manifests/v3_1/external_inventory_finite_sky/complete.json'
    legacy = json.loads(legacy_record.read_text())['inputs']
    legacy_name, = [name for name in legacy if Path(name).name == 'allStar-dr17-synspec_rev1.fits']
    legacy_path = verified(Path(legacy_name), legacy[legacy_name])
    old = fitsio.read(str(legacy_path), ext=1, columns=['APOGEE_ID', 'GAIAEDR3_SOURCE_ID'])
    ds, sp = [read_apogee(paths[k]) for k in ('distance', 'stellar')]
    a, b, counts = inventory_join(ds, sp, old['GAIAEDR3_SOURCE_ID'][old['GAIAEDR3_SOURCE_ID'] > 0],
                                   np.unique(text(old['APOGEE_ID'])[text(old['APOGEE_ID']) != '']))
    ids = ds['gaia_dr3_source_id'][a]
    ra, dec = ds['ra'][a].astype(float), ds['dec'][a].astype(float)
    sky, spec_sky = qualified_sky(ra, dec), qualified_sky(sp['ra'][b], sp['dec'][b])
    valid = sky['valid_coordinates'] & spec_sky['valid_coordinates']
    close = np.zeros(len(a), bool)
    close[valid] = np.linalg.norm(unit_vectors(ra[valid], dec[valid])
        - unit_vectors(sp['ra'][b[valid]], sp['dec'][b[valid]]), axis=1) <= 2*np.sin(np.deg2rad(1/7200))
    highlat = valid & close & (abs(sky['latitude']) > 30)

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
    out = ROOT/'data/v3_2/reference_inventory'
    out.mkdir(parents=True, exist_ok=False)
    # This is a match-request index, not a scientific sample. Native TYPE has
    # yet to be observed. Scientific export occurs only after native admission.
    atomic_npz(out/'requests.npz', identity=ids[field_rows], ra=ra[field_rows], dec=dec[field_rows],
               cell=sky['cell'][field_rows], sdss_id=ds['sdss_id'][a[field_rows]],
               distance_row=a[field_rows], stellar_row=b[field_rows])
    csv_rows(out/'history.csv', rows)
    counts.update(distance_rows=len(ds['sdss_id']), stellar_rows=len(sp['sdss_id']),
                  distinct_joint_ids=len(ids), coordinate_agreement=int(close.sum()),
                  high_latitude=int(highlat.sum()), prior_id_excluded=int((highlat & exposed).sum()),
                  old_sky_excluded=int((highlat & ~exposed & np.isin(sky['cell'], list(occupied))).sum()),
                  footprint_excluded=int((highlat & ~exposed & ~np.isin(sky['cell'], list(occupied)) & field).sum()),
                  known_system_excluded=int(group['external_any'].sum()),
                  match_requests=len(field_rows), scientific_admissions=0)
    save_json(out/'counts.json', counts)
    inputs = list(paths.values()) + history + group_paths + configs + [
        legacy_path, legacy_record, r1, typed_path, Path(__file__).resolve(),
        ROOT/'docs/v3_2_reference_intake.md', ROOT/'manifests/v3_1/external_history/complete.json',
        ROOT/'scripts/v3_1_external_history.py', ROOT/'src/photdis/v3_1/sky.py',
        ROOT/'src/photdis/holdouts.py']
    save_json(out/'complete.json', dict(
        inputs={str(p): sha(p) for p in inputs}, outputs={p.name: sha(p) for p in out.iterdir()},
        job=os.getenv('SLURM_JOB_ID'), native_TYPE_unverified=len(field_rows), outcomes_read=False,
        archive_lineage={k: {f'{release}/{apred}': n for (release, apred), n in
                             Counter(zip(text(t['release']), text(t['apred']))).items()}
                         for k, t in [('distance', ds), ('stellar', sp)]},
        scientific_sample_exported=False))


if __name__ == '__main__':
    main()

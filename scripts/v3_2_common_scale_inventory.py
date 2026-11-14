#!/usr/bin/env python3
"""Inventory previously unscored HST-scale system photometry without scale values."""
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import zipfile

import fitsio
import numpy as np
from scipy.spatial import cKDTree

from photdis.assembly_v1_4 import unit_vectors, uppercase_fields
from photdis.holdouts import GC_ARCHIVE_ALIASES
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3_1.sky import qualified_sky
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_native import NEW_SKY
from v3_1_external_history import HISTORY
import v3_1_external_native as native
from v3_1_freeze_roles import history_columns
from v3_2_reference_inventory import footprint
from v3_2_tilt_inputs import verified

# These systems have prior cluster work or v3.1 availability/score exposure.
DOCUMENTED = {'NGC_288', 'NGC_1261', 'NGC_1851', 'NGC_7492', 'NGC_5904_M_5',
              'NGC_6341_M_92', 'NGC_7078_M_15'}


def published_systems(table, previous):
    # The availability query uses only the keys of the published methods.
    methods = {r['name']: 'hst_kinematic' in r['displayed_methods'] for r in table['rows']}
    rows = [r for r in table['membership_mapping'] if methods.get(r['reference_name'], False)]
    sky = qualified_sky([r['ra'] for r in rows], [r['dec'] for r in rows])
    for row, latitude in zip(rows, sky['latitude']):
        name = row['member_name']
        history = previous[name]['mws_member_spectra']
        status = ('documented_history' if name in DOCUMENTED or history else
                  'outside_center_buffer' if abs(latitude) < 28 or row['dec'] > 35 else 'targeted')
        yield dict(system=name, group=int(row['group']), latitude=float(latitude),
                   published_label_catalog_overlap=history, preselection=status)


def main():
    require_compute()
    out = ROOT/'data/v3_2/common_scale_inventory'
    if out.exists():
        raise FileExistsError(out)
    manifests = ROOT/'manifests'
    reference = manifests/'kgiant_bv21_reference_v1.json'
    previous = manifests/'giant_parameter_inventory_v1.json'
    archive_record = manifests/'holdout_reference_manifest.json'
    archive_name = 'vasiliev_baumgardt_2021_members.zip'
    archive_path = verified(ROOT/'data/reference'/archive_name,
                            json.loads(archive_record.read_text())['sources'][archive_name]['sha256'])
    table = list(published_systems(json.loads(reference.read_text()), json.loads(previous.read_text())['systems']))
    targeted = [r for r in table if r['preselection'] == 'targeted']
    members = []
    with zipfile.ZipFile(archive_path) as archive:
        for sys in targeted:
            name = sys['system']
            with archive.open('catalogues/'+GC_ARCHIVE_ALIASES.get(name, name)+'.txt') as f:
                for line in io.TextIOWrapper(f):
                    if line.startswith('#') or not line.strip():
                        continue
                    x = line.split()
                    if float(x[-1]) > .9:
                        members.append((int(x[0]),float(x[1]),float(x[2]),sys['group']))
    member = np.array(members, dtype=[('id','i8'),('ra','f8'),('dec','f8'),('group','i8')])
    ids = member['id']
    if (ids <= 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique exact positive published member IDs required')
    role_marker = manifests/'v3_1/validation_roles/complete.json'
    role = json.loads(role_marker.read_text())
    history = [verified(ROOT/path, role['inputs'][path]) for path, _ in HISTORY]
    history += [verified(ROOT/'data/v3_1/validation_roles'/p, role['outputs'][p])
                for p in ('field_photometry.npz','common_photometry.npz')]
    exposed = set()
    historical_rows = []
    imaging_ids = []
    for path in history:
        old = history_columns(path)
        sid = old.get('identity', old.get('source_id'))
        hit = np.isin(ids, sid[sid > 0])
        exposed.update(member['group'][hit].tolist())
        historical_rows.extend(dict(path=str(path.relative_to(ROOT)), group=r['group'],
                                    member_overlap=int((hit & (member['group'] == r['group'])).sum()))
                               for r in targeted)
        imaging_ids.append(np.r_[old['ls_id'][old['ls_id'] > 0],-sid[sid < 0]])
    catalog = native.SOUTH/'survey-bricks-dr11-south.fits.gz'
    bricks = uppercase_fields(fitsio.read(str(catalog), columns=['BRICKNAME','RA','DEC']))
    eligible = ~np.isin(member['group'], list(exposed))
    native._IDS, native._OUT = ids, out
    nearest = cKDTree(unit_vectors(bricks['RA'], bricks['DEC'])).query_ball_point(
        unit_vectors(member['ra'][eligible], member['dec'][eligible]), 2*np.sin(np.deg2rad(.4)/2))
    assignments = {}
    for row, neighbors in zip(np.flatnonzero(eligible), nearest):
        for j in neighbors:
            assignments.setdefault(j, []).append(row)
    names = np.char.strip(bricks['BRICKNAME'].astype('U'))
    tasks = [(names[j], np.array(ix)) for j, ix in sorted(assignments.items())]
    (out/'bricks').mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('fork')) as pool:
        records = list(pool.map(native.brick, tasks))
    save_json(out/'native_sources.json', records)
    pieces = [load(verified(out/'bricks'/f"{r['brick']}.npz",r['sha256']))
              for r in records if r.get('matched_primary')]
    configs = []
    bounds = list(NEW_SKY)
    for p in sorted((ROOT/'configs').glob('*.json')):
        config = json.loads(p.read_text())
        if 'bounds_deg' in config:
            configs.append(p)
            bounds.append(config['bounds_deg'])
    measured = {}
    outputs = ['native_sources.json','systems.csv','history.csv']
    if pieces:
        data = {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        gid, lid = data['identity'], data['ls_id']
        match = unique_join(gid, ids)
        if (match < 0).any():
            raise ValueError('Unexpected native member identity')
        group = member['group'][match]
        exposed.update(group[np.isin(lid, np.concatenate(imaging_ids))].tolist())
        psf = (data['TYPE'] == 'PSF') & (data['RELEASE'] == 11010)
        unique = (unique_join(gid,gid) >= 0) & (unique_join(lid,lid) >= 0)
        same_domain = psf & unique & data['photometric_entry']
        seen_sky = footprint(data['RA'],data['DEC'],bounds)
        exposed.update(group[seen_sky].tolist())
        admitted = same_domain & ~np.isin(group, list(exposed))
        for g in np.unique(group):
            s = group == g
            measured[int(g)] = dict(exact_primary=int(s.sum()),
                nonPSF=int((s & (data['TYPE'] != '') & (data['TYPE'] != 'PSF')).sum()),
                missing_TYPE=int((s & (data['TYPE'] == '')).sum()),
                duplicate=int((s & ~unique).sum()),
                native_psf=int((s & psf & unique).sum()),
                photometric_domain=int((s & same_domain).sum()),
                prior_footprint=int((s & seen_sky).sum()), admitted=int((s & admitted).sum()))
        atomic_npz(out/'photometry.npz', **{k:v[admitted] for k,v in data.items()}, scale_system=group[admitted])
        outputs.append('photometry.npz')
    for r in table:
        g = r['group']
        r.update(strict_members=int((member['group'] == g).sum()) if r['preselection']=='targeted' else None,
                 prior_exposure=g in exposed,
                 **measured.get(g, dict(exact_primary=0, nonPSF=0, missing_TYPE=0, duplicate=0,
                       native_psf=0, photometric_domain=0, prior_footprint=0, admitted=0)))
    csv_rows(out/'systems.csv', table)
    csv_rows(out/'history.csv', historical_rows)
    inputs = history+configs+[catalog, reference, previous, archive_record, archive_path,
                              role_marker, Path(__file__).resolve(), ROOT/'scripts/v3_1_external_native.py',
                              ROOT/'scripts/v3_1_freeze_roles.py', ROOT/'scripts/v3_0_native.py']
    save_json(out/'complete.json',dict(inputs={str(p):sha(p) for p in inputs},
        outputs={name:sha(out/name) for name in outputs}, job=os.getenv('SLURM_JOB_ID'),
        targeted_systems=len(targeted), targeted_bricks=len(tasks),
        missing_TYPE_brick_matches=sum(r.get('missing_TYPE',0) for r in records),
        admitted_systems=sum(r['admitted']>0 for r in table),
        admitted_members=sum(r['admitted'] for r in table), all_admitted_PSF=True,
        scale_values_exported=False, predictions_read=False, roles_frozen=False))


if __name__ == '__main__':
    main()

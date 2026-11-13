#!/usr/bin/env python3
"""Targeted published-member/native availability with whole-system history guards."""
import concurrent.futures
import io
import json
import multiprocessing as mp
import os
import zipfile
import fitsio
import numpy as np
from scipy.spatial import cKDTree
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.holdouts import GC_ARCHIVE_ALIASES
from photdis.assembly_v1_4 import unit_vectors, uppercase_fields
from photdis.reference_v1_4 import unique_join
from v3_1_external_history import HISTORY, identity_position
import v3_1_external_native as native


def main():
    require_compute(); out = ROOT/'data/v3_1/common_scale_native'; out.mkdir(exist_ok=True)
    archive_path = ROOT/'data/reference/vasiliev_baumgardt_2021_members.zip'
    scale_path = ROOT/'manifests/kgiant_bv21_reference_v1.json'
    prior_path = ROOT/'manifests/giant_parameter_inventory_v1.json'
    bricks_path = native.SOUTH/'survey-bricks-dr11-south.fits.gz'
    configs = [p for p in sorted((ROOT/'configs').glob('*.json')) if 'bounds_deg' in json.loads(p.read_text())]
    paths = [archive_path, scale_path, prior_path, bricks_path, ROOT/'docs/v3_1_common_scale_protocol.md',
        ROOT/'scripts/v3_1_common_scale_native.py', ROOT/'scripts/v3_1_external_native.py',
        ROOT/'scripts/v3_1_external_history.py', ROOT/'scripts/v3_0_native.py']+configs+[ROOT/name for name, _ in HISTORY]
    inputs = {str(p): sha(p) for p in paths}; freeze = out/'inputs.json'; marker = out/'complete.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze, inputs)
    if marker.exists():
        done = json.loads(marker.read_text()); assert done['inputs'] == inputs
        for name, digest in done['outputs'].items(): assert sha(out/name) == digest
        return
    names = ['NGC_1261', 'NGC_1851', 'NGC_7492']; reference = json.loads(scale_path.read_text()); old_inventory = json.loads(prior_path.read_text())
    mapping = {r['member_name']: r for r in reference['membership_mapping']}
    rows = []; membership_counts = {}
    with zipfile.ZipFile(archive_path) as archive:
        for system, name in enumerate(names):
            assert old_inventory['systems'][name]['mws_member_spectra'] == 0
            count = 0
            with archive.open('catalogues/'+GC_ARCHIVE_ALIASES.get(name, name)+'.txt') as stream:
                for line in io.TextIOWrapper(stream):
                    if not line.strip() or line.startswith('#'): continue
                    v = line.split()
                    if float(v[-1]) > .9:
                        rows.append((int(v[0]), float(v[1]), float(v[2]), float(v[-1]), system)); count += 1
            membership_counts[name] = count
    member = np.array(rows, dtype=[('identity', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('probability', 'f8'), ('system', 'i2')])
    assert len(np.unique(member['identity'])) == len(member) and (member['identity'] > 0).all()
    seen = np.zeros(len(member), bool); seen_system = np.zeros(len(names), bool); history_rows = []
    for path, lineage in HISTORY:
        old, _ = identity_position(ROOT/path); match = np.isin(member['identity'], old[old > 0]); seen |= match
        seen_system[np.unique(member['system'][match])] = True
        history_rows.append(dict(path=path, lineage=lineage, member_overlap=int(match.sum())))
    native._IDS = member['identity']; native._OUT = out; (out/'bricks').mkdir(exist_ok=True)
    b = uppercase_fields(fitsio.read(str(bricks_path), columns=['BRICKNAME', 'RA', 'DEC']))
    near = cKDTree(unit_vectors(b['RA'], b['DEC'])).query_ball_point(unit_vectors(member['RA'], member['DEC']), 2*np.sin(np.deg2rad(.4)/2))
    owners = {}
    for i, neighbors in enumerate(near):
        for j in neighbors: owners.setdefault(int(j), []).append(i)
    tasks = [(str(b['BRICKNAME'][j].decode() if isinstance(b['BRICKNAME'][j], bytes) else b['BRICKNAME'][j]).strip(), np.array(ix)) for j, ix in sorted(owners.items())]
    print('COMMON SCALE NATIVE', membership_counts, len(tasks), 'bricks', flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('fork')) as pool:
        records = list(pool.map(native.brick, tasks))
    save_json(out/'native_sources.json', records)
    pieces = [load(out/'bricks'/f"{r['brick']}.npz") for r in records if r.get('matched_primary', 0)]
    if not pieces: raise RuntimeError('No native matches; acquisition/checkpoints retained')
    t = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    assert len(np.unique(t['ls_id'])) == len(t['ls_id'])
    j = unique_join(t['identity'], member['identity']); assert (j >= 0).all()
    source = member[j]; _, inv, counts = np.unique(t['identity'], return_inverse=True, return_counts=True)
    unique = counts[inv] == 1; psf = (t['TYPE'] == 'PSF') & (t['RELEASE'] == 11010)
    footprints = np.zeros(len(t['identity']), bool)
    for path in configs:
        lo, de, hi, dh = json.loads(path.read_text())['bounds_deg']
        inside = ((t['RA'] >= lo) & (t['RA'] <= hi)) if lo <= hi else ((t['RA'] >= lo) | (t['RA'] <= hi))
        footprints |= inside & (t['DEC'] >= de) & (t['DEC'] <= dh)
    # Include the two once-opened v3 new-sky footprints explicitly.
    footprints |= ((t['RA'] >= 10) & (t['RA'] <= 15) | (t['RA'] >= 350) & (t['RA'] <= 355)) & (t['DEC'] >= -45) & (t['DEC'] <= -40)
    selected = unique & psf & t['photometric_entry'] & ~footprints & ~seen[j] & ~seen_system[source['system']]
    summaries = []
    for system, name in enumerate(names):
        use = source['system'] == system
        summaries.append(dict(system=name, strict_published_members=membership_counts[name], exact_native=int(use.sum()),
            nonPSF=int((use & (t['TYPE'] != '') & (t['TYPE'] != 'PSF')).sum()), missing_TYPE=int((use & (t['TYPE'] == '')).sum()),
            duplicate_source=int((use & ~unique).sum()), native_PSF=int((use & psf & unique).sum()),
            photometric_domain=int((use & psf & unique & t['photometric_entry']).sum()),
            historical_member_exposure=bool(seen_system[system]), historical_footprint=int((use & footprints).sum()),
            admitted=int((use & selected).sum()), cells=len(np.unique(t['cell'][use & selected]))))
    atomic_npz(out/'photometry.npz', **{k: v[selected] for k, v in t.items()}, scale_system=source['system'][selected])
    atomic_npz(out/'membership.npz', identity=t['identity'][selected], TYPE=t['TYPE'][selected],
        scale_system=source['system'][selected], probability=source['probability'][selected])
    scales = []
    for name in names:
        reference_name = mapping[name]['reference_name']; r = next(x for x in reference['rows'] if x['name'] == reference_name)
        method = 'hst_kinematic' if name != 'NGC_7492' else 'literature'
        scales.append(dict(system=name, method=method, reference=r['methods'][method],
            covariance='One common distance per system; HST common-method3percent sensitivity separately; no independent member errors',
            role='availability only; final definition/method/prediction freeze required'))
    save_json(out/'scales.json', scales); csv_rows(out/'admission.csv', summaries); csv_rows(out/'history.csv', history_rows)
    save_json(marker, dict(inputs=inputs, outputs={name: sha(out/name) for name in ['photometry.npz', 'membership.npz', 'scales.json', 'admission.csv', 'history.csv', 'native_sources.json']},
        job=os.getenv('SLURM_JOB_ID'), targeted_bricks=len(tasks), systems=summaries, admitted=int(selected.sum()),
        missing_TYPE=sum(r.get('missing_TYPE', 0) for r in records), final_roles_frozen=False, predictions_read=False,
        scope='New strict-member native availability only. Published membership depends on Gaia; external common distances have shared scale/method uncertainty. Empty systems or historical exposure are not repaired by relaxed cuts.'))
    print('COMMON SCALE COMPLETE', summaries, flush=True)


if __name__ == '__main__': main()

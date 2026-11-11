#!/usr/bin/env python3
"""Photometric-first NGC288 independent-scale layer; association waits for E."""
import argparse
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from v3_0_native import read_native, native_payload
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.v3.data import stable_rank

D = ROOT/'data/v3_0'


def main():
    require_compute(); parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['photometry', 'associate']); args = parser.parse_args()
    cfg_path = ROOT/'configs/v3_0_independent_scale.json'; cfg = json.loads(cfg_path.read_text())
    destination = D/'independent_scale'; destination.mkdir(exist_ok=True)
    source = ROOT/'data/reference'
    catalog_path = source/'vasiliev_baumgardt_2021_clusters.txt'
    catalog = np.genfromtxt(catalog_path, names=True, dtype=None, encoding='utf-8')
    names = np.array([str(s).replace('_', '').replace(' ', '') for s in catalog['Name']])
    center = catalog[np.flatnonzero(names == cfg['system'])[0]]
    ra, dec = float(center['ra']), float(center['dec'])
    output = destination/'photometry.npz'; marker = destination/'photometry.json'
    if args.stage == 'photometry':
        if marker.exists():
            record = json.loads(marker.read_text()); assert record['config_sha256'] == sha(cfg_path)
            assert record['sha256'] == sha(output); return
        sweep_root = Path(json.loads((ROOT/'configs/baseline_v1_4.json').read_text())['sweep_root'])
        sweep = sweep_root/'sweep-010m030-015m025.fits'
        raw = read_native(sweep)
        cosangle = (np.sin(np.deg2rad(raw['DEC']))*np.sin(np.deg2rad(dec))+
            np.cos(np.deg2rad(raw['DEC']))*np.cos(np.deg2rad(dec))*np.cos(np.deg2rad(raw['RA']-ra)))
        angle = np.rad2deg(np.arccos(np.clip(cosangle, -1, 1)))
        region = angle <= cfg['photometric_radius_deg']
        assert ra-cfg['photometric_radius_deg']/np.cos(np.deg2rad(dec)) > 10 and ra+cfg['photometric_radius_deg']/np.cos(np.deg2rad(dec)) < 15
        assert dec-cfg['photometric_radius_deg'] > -30 and dec+cfg['photometric_radius_deg'] < -25
        t = native_payload(raw[region], 'DECam_DR11'); t['radius_deg'] = angle[region]
        admitted = t['photometric_entry']; rows = np.flatnonzero(admitted)
        rows = rows[np.argsort(stable_rank(t['identity'][rows], cfg['photometric_sampling_seed']))[:cfg['maximum_photometric_sample']]]
        result = {k: v[rows] for k, v in t.items()}
        result['field'] = np.full(len(rows), 102, 'i2'); result['design_weight'] = np.full(len(rows), admitted.sum()/len(rows))
        assert (result['TYPE'] == 'PSF').all() and len(np.unique(result['identity'])) == len(rows)
        # Verify isolation from every evidence-fitting role before any association.
        with np.load(D/'references.npz') as z:
            role, ids = z['role'], z['identity']
        overlaps = {str(r): int(np.isin(result['identity'], ids[role == r]).sum()) for r in (1, 2, 3, 4)}
        if any(overlaps.values()): raise RuntimeError('Independent-system identity isolation failed: '+str(overlaps))
        atomic_npz(output, **result)
        save_json(marker, dict(config_sha256=sha(cfg_path), sha256=sha(output),
            region_rows=int(region.sum()), admitted=int(admitted.sum()), sampled=len(rows),
            nonPSF=int((~t['psf'] & (t['TYPE'] != '')).sum()), missing_TYPE=int((t['TYPE'] == '').sum()),
            photometric_failure=int((t['psf'] & ~admitted).sum()), role_overlaps=overlaps,
            membership_outcomes_opened=False, scale_used_in_fitting=False,
            center_ra=ra, center_dec=dec, source_sweep=str(sweep),
            catalog_sha256=sha(catalog_path), job=os.getenv('SLURM_JOB_ID')))
        print('INDEPENDENT PHOTOMETRY', len(rows), 'of', int(admitted.sum()), 'no memberships/predictions opened', flush=True)
        return
    method = D/'method_freeze.json'
    if not method.exists(): raise RuntimeError('Final method freeze required before independent membership evaluation')
    t = load(output); record = json.loads(marker.read_text()); assert sha(output) == record['sha256']
    association = destination/'association.json'
    if association.exists():
        old = json.loads(association.read_text())
        assert old['method_freeze_sha256'] == sha(method) and old['config_sha256'] == sha(cfg_path)
        assert old['input_sha256'] == sha(output)
        return
    archive_path = source/'vasiliev_baumgardt_2021_members.zip'
    with zipfile.ZipFile(archive_path) as archive:
        paths = [p for p in archive.namelist() if p.split('/')[-1].replace('_', '').replace('.txt', '') == cfg['system']]
        assert len(paths) == 1
        with archive.open(paths[0]) as stream:
            records = [line.split() for line in io.TextIOWrapper(stream) if line.strip() and not line.startswith('#')]
    ids = np.array([int(r[0]) for r in records], 'i8'); probability = np.array([float(r[-1]) for r in records])
    match = unique_join(t['source_id'], ids); p = np.full(len(match), np.nan); found = match >= 0
    p[found] = probability[match[found]]
    atomic_npz(destination/'membership.npz', identity=t['identity'], TYPE=t['TYPE'],
        membership_probability=p, membership_available=found,
        selected_member=found & (p >= cfg['membership_probability_minimum']))
    save_json(destination/'association.json', dict(method_freeze_sha256=sha(method),
        input_sha256=sha(output), archive_sha256=sha(archive_path), config_sha256=sha(cfg_path),
        n=len(match), matched=int(found.sum()), selected=int((p >= cfg['membership_probability_minimum']).sum()),
        membership_conditioning=cfg['membership_source'], distance_scale=cfg,
        job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__': main()

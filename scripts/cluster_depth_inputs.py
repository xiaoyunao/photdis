#!/usr/bin/env python3
"""Recover the archived native identity roster and its imaging metadata."""
import json
from pathlib import Path
import zipfile

from astropy.coordinates import SkyCoord
from fitsio import FITS
import numpy as np

import environment_pilot as pilot
from photdis.stage1 import legacy_ids, require_compute, tile_bounds
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_5_dr11_pilot import in_tile

KEY = 'cluster_depth_20261120'
CONFIG = ROOT/'configs'/f'{KEY}.json'
C = json.loads(CONFIG.read_text())
OLD = ROOT/'data/faint_signed_20261120'
OUT = ROOT/'data'/KEY
OLD_CONFIG = ROOT/'configs/faint_signed_20261120.json'


def imaging(ids, paths, release):
    metrics = [f'{x}_{b}' for x in ('FRACMASKED','FRACIN','NOBS','ALLMASK') for b in 'GRZ']
    used = np.zeros(len(ids),int)
    result = {k:np.full(len(ids),np.nan) for k in metrics}
    for filename in paths:
        path = Path(filename)
        with FITS(str(path)) as f:
            raw = f[1].read(columns=['RELEASE','BRICKID','OBJID','RA','DEC','TYPE',*metrics])
        j = pilot.unique_join(legacy_ids(raw),ids)
        take = (j>=0)&in_tile(raw['RA'],raw['DEC'],tile_bounds(path))&(raw['RELEASE']==release)
        j = j[take]
        assert (np.char.strip(raw['TYPE'][take].astype(str))=='PSF').all()
        np.add.at(used,j,1)
        for k in metrics:
            result[k][j] = raw[k][take]
    np.testing.assert_array_equal(used,np.ones(len(ids)))
    return result


def member_probabilities(source_id,name,archive):
    prefix = '_'.join(name.split('_')[:2])
    matches = [f for f in archive.namelist() if f.startswith('catalogues/') and
               '_'.join(Path(f).stem.split('_')[:2])==prefix]
    assert len(matches)==1
    rows = [ln.split() for ln in archive.read(matches[0]).decode().splitlines()
            if ln.strip() and not ln.startswith('#')]
    sid = np.array([int(r[0]) for r in rows],dtype='i8')
    index = pilot.unique_join(source_id,sid)
    assert (index>=0).all()
    pr = np.array([float(r[-1]) for r in rows])[index]
    assert (pr>.9).all()
    return pr


def main():
    require_compute()
    OUT.mkdir(parents=True,exist_ok=True)
    seal = OLD/'results/cluster_freeze.json'
    frozen = json.loads(seal.read_text())
    assert frozen['config']==sha(OLD_CONFIG)
    source = ROOT/'data/reference/vasiliev_baumgardt_2021_members.zip'
    assert frozen['member_archive']==sha(source)
    prior = json.loads(OLD_CONFIG.read_text())['cluster_candidates']
    pilot.C = dict(pilot.C,field_radius_deg=.5)
    rows = []
    receipts = {}
    with zipfile.ZipFile(source) as archive:
        for key in C['systems']:
            ids = np.asarray(frozen['admitted'][key],dtype='i8')
            old_path = OLD/(key+'_fixed_pdfs.npz')
            pdf = load(old_path)
            np.testing.assert_array_equal(pdf['ls_id'], ids)
            loc = prior[key]['galactic']
            parts,ledger,paths = pilot.field_rows([key,*loc])
            original = pilot.join(parts)
            pos = pilot.unique_join(ids,original['ls_id'])
            assert (pos>=0).all()
            t = pilot.subset(original,pos)
            np.testing.assert_array_equal(t['ls_id'],ids)
            assert (t['TYPE']=='PSF').all() and (pdf['TYPE']=='PSF').all()
            center = SkyCoord(*loc,unit='deg',frame='galactic').icrs
            coord = SkyCoord(t['RA'],t['DEC'],unit='deg')
            radius=coord.separation(center).arcmin
            t.update(imaging(ids,paths,pilot.settings()['south']['release']))
            t['radius']=radius
            t['member_p']=member_probabilities(t['source_id'],key,archive)
            out=OUT/(key+'_native.npz')
            atomic_npz(out,**t)
            receipts[key]=dict(native=sha(out),pdf=sha(old_path),sweeps=paths,roster_sha=sha(seal))
            for r in ledger:
                rows.append(dict(**r,eligible_frozen=len(ids)))
            print(key,len(ids),'same native PSF rows recovered',flush=True)
    csv_rows(OUT/'admission.csv',rows)
    save_json(OUT/'inputs.json',dict(config=sha(CONFIG),member_archive=sha(source),
                                    files=receipts))


if __name__=='__main__':
    main()

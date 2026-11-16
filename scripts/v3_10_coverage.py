#!/usr/bin/env python3
"""Measure footprint, masks and optical depth for the working sample."""
import argparse
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np

from photdis.photometry import CLEAN_REJECT_MASKBITS
from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_5_dr11_area import BANDS, NSIDE, random_memmap, tile_map
from v3_9_catalog import OUT as EXPANSION, settings
from v3_9_seam import BOUNDARY

DEST = ROOT/'data/v3_10/coverage'
M92 = (259.28076,43.13594)
ANNULI = ((.08,.35),(.65,1.20))


def m92_zones(ra,dec):
    a,b = np.deg2rad([ra,dec])
    l,d = np.deg2rad(M92)
    theta = np.rad2deg(np.arccos(np.clip(np.sin(b)*np.sin(d)+np.cos(b)*np.cos(d)*np.cos(a-l),-1.,1.)))
    holdout = (np.floor(ra/10).astype(int)+3*np.floor((dec+90)/10).astype(int))%5==4
    return [(theta>=lo)&(theta<hi)&holdout for lo,hi in ANNULI]


def run(side):
    root = Path(settings()[side]['sweep_root']).parent.parent/'randoms'
    path = root/f'randoms-{side}-1-0.fits'
    raw,density,names = random_memmap(path)
    marker = EXPANSION/side/'inventory.json'
    inventory = json.loads(marker.read_text())
    native = settings()[side]['release']
    sweeps = [s for group in inventory['groups'] for s in group]
    tiles = tile_map(sweeps)
    npix = hp.nside2npix(NSIDE)
    census = np.zeros((4,npix),'i8')
    m92 = np.zeros((4,2),'i8')
    reject = sum(1<<b for b in CLEAN_REJECT_MASKBITS)
    flux = 10**((22.5-20.)/2.5)
    depth_minimum = (2.5/np.log(10)/.1/flux)**2
    expected = ('N' if side=='north' else 'S').encode()
    for start in range(0,len(raw),300_000):
        block = raw[start:start+300_000]
        t = {key:block[name] for key,name in names.items()}
        ra,dec = t['RA'],t['DEC']
        b = galactic(ra,dec)[1]
        dcut = abs(b)>30 if side=='north' else abs(b)<=30
        owner = ((dec>=BOUNDARY)&(b>0))==(side=='north')
        ij = tiles[(ra%360).astype(int),np.minimum((dec+90).astype(int),179)]
        geo = owner&dcut&(ij>=0)&(t['EBV']>=0)&(t['EBV']<.03)
        coverage = geo&(t['RELEASE']==native)&(np.char.strip(t['PHOTSYS'])==expected)
        coverage &= np.logical_and.reduce([t[f'NOBS_{k}']>0 for k in BANDS])
        unmasked = coverage&((t['MASKBITS'].astype('i8')&reject)==0)
        deep = unmasked&np.logical_and.reduce([t[f'PSFDEPTH_{k}']>=depth_minimum for k in BANDS])
        pixel = hp.ang2pix(NSIDE,ra,dec,lonlat=True,nest=True)
        zones = m92_zones(ra,dec) if side=='north' else ()
        for i,level in enumerate((geo,coverage,unmasked,deep)):
            census[i] += np.bincount(pixel[level],minlength=npix)
            for j,mask in enumerate(zones):
                m92[i,j] += (level&mask).sum()
    if not census[2].any():
        raise ValueError('The registered random sample has no native unmasked footprint')
    fields = ('geometry_deg2','grz_deg2','unmasked_deg2','grz_depth20_deg2')
    area = census/density
    d = DEST/side
    d.mkdir(parents=True,exist_ok=True)
    rows = [dict(pixel=i,**dict(zip(fields,area[:,i]))) for i in np.flatnonzero(census[0])]
    csv_rows(d/'healpix_area.csv',rows)
    csv_rows(d/'m92_area.csv',[dict(annulus=i,**dict(zip(fields,m92[:,i]/density))) for i in range(2)])
    atomic_npz(d/'area.npz',area_deg2=area,m92_area_deg2=m92/density)
    stat = path.stat()
    save_json(d/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),input_path=str(path),
        input_size=stat.st_size,input_mtime_ns=stat.st_mtime_ns,sweep_inventory_sha256=sha(marker),
        density=density,**dict(zip(fields,area.sum(axis=1))),
        outputs={p.name:sha(p) for p in d.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('side',choices=('north','south'))
    run(p.parse_args().side)

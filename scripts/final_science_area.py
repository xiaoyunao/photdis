#!/usr/bin/env python3
"""Apply native ownership and depth to the actual Legacy Surveys randoms."""
import argparse
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np

from photdis.photometry import CLEAN_REJECT_MASKBITS
from photdis.stage1 import galactic, require_compute
from v2_1_near import atomic_npz, csv_rows, save_json, sha
from v3_5_dr11_area import BANDS, random_memmap, tile_map
from v3_9_catalog import OUT as EXPANSION, settings
from v3_9_seam import BOUNDARY
from final_science_catalog import C, CONFIG, DEST

DEPTHS = (19.4,20.,21.,22.,23.)
FIELDS = ('coverage','unmasked',*(f'grz_{v:g}' for v in DEPTHS))
BANDS_LABEL = ('low','high')


def run(side, *, nside=C['sky_nside'], destination=DEST/'area'):
    root = Path(settings()[side]['sweep_root']).parent.parent/'randoms'
    path = root/f'randoms-{side}-1-0.fits'
    before = path.stat()
    raw,density,names = random_memmap(path)
    inventory = EXPANSION/side/'inventory.json'
    meta = json.loads(inventory.read_text())
    tiles = tile_map([s for group in meta['groups'] for s in group])
    npix = hp.nside2npix(nside)
    counts = np.zeros((2,len(FIELDS),npix),'i8')
    reject = sum(1<<b for b in CLEAN_REJECT_MASKBITS)
    depths = (2.5/.1/np.log(10)/10**((22.5-np.array(DEPTHS))/2.5))**2
    release = settings()[side]['release']
    expected = (b'N' if side=='north' else b'S')
    for start in range(0,len(raw),250_000):
        data = raw[start:start+250_000]
        t = {k:data[v] for k,v in names.items()}
        ra,dec = t['RA'],t['DEC']
        _,b = galactic(ra,dec)
        geo = ((dec>=BOUNDARY)&(b>0))==(side=='north')
        geo &= (t['EBV']>=0)&(t['EBV']<.03)
        geo &= tiles[(ra%360).astype(int),np.minimum((dec+90).astype(int),179)]>=0
        cov = geo&(t['RELEASE']==release)&(np.char.strip(t['PHOTSYS'])==expected)
        cov &= np.logical_and.reduce([t[f'NOBS_{k}']>0 for k in BANDS])
        if not cov.any():
            continue
        j = np.flatnonzero(cov)
        lo = abs(b[j])<=30
        mask = (t['MASKBITS'][j].astype('i8')&reject)==0
        dp = np.minimum.reduce([t[f'PSFDEPTH_{k}'][j] for k in BANDS])
        pix = hp.ang2pix(nside,ra[j],dec[j],lonlat=True,nest=True)
        for zone,region in enumerate((lo,~lo)):
            for i,level in enumerate((region,region&mask,*(region&mask&(dp>=v) for v in depths))):
                counts[zone,i] += np.bincount(pix[level],minlength=npix)
    area = counts/density
    if not area[1,0].any():
        raise ValueError('No eligible randoms in high latitude native footprint')
    after = path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Random catalog changed during analysis')
    dest = destination/side
    dest.mkdir(exist_ok=True,parents=True)
    atomic_npz(dest/'area.npz',effective_deg2=area)
    csv_rows(dest/'area_totals.csv',[dict(side=side,latitude=label,**dict(zip(FIELDS,a.sum(axis=1))))
                                  for label,a in zip(BANDS_LABEL,area)])
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),config=sha(CONFIG),
        inventory=sha(inventory),random_path=str(path),random_bytes=before.st_size,
        random_mtime_ns=before.st_mtime_ns,rows=len(raw),density=density,nside=nside,
        outputs={p.name:sha(p) for p in (dest/'area.npz',dest/'area_totals.csv')}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('side',choices=('north','south'))
    run(parser.parse_args().side)

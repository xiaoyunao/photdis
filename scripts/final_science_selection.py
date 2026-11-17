#!/usr/bin/env python3
"""Measure which native bright populations the working selection keeps on the sky."""
import json
import os

import healpy as hp
import numpy as np

from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_10_science import working_index
from final_science_area import FIELDS
from final_science_catalog import C, DEST

RANGES = ((0.,.5),(.5,.8),(.8,1.4))
MAG = (14.,16.,18.,20.)
LAT = (0.,15.,30.,40.,50.,60.,75.,90.01)
LEVELS = ('bright_parent','strict_grz','qrf_supported','dwarf_score','working_pass','science_2_5')


def run(side, inputs, cells, profiles):
    _, t = working_index(side,inputs)
    marker = DEST/'area'/side/'complete.json'
    meta = json.loads(marker.read_text())
    p = verified(marker.with_name('area.npz'),meta['outputs']['area.npz'])
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    nside = C['sky_nside']
    n = hp.nside2npix(nside)
    ar = read_columns(p,('effective_deg2',))['effective_deg2'][int(side=='north')]
    area = ar[FIELDS.index('grz_21')]
    clean_area = (ar[1]>0)&(area>=.9*ar[1])&(area>=.5)
    ra,dec = hp.pix2ang(nside,np.arange(n),lonlat=True,nest=True)
    _,bc = galactic(ra,dec)
    k = hp.ang2pix(nside,t['RA'],t['DEC'],lonlat=True,nest=True)
    score = t['state']
    distance = t['qrf_median_kpc']
    c = t['g0']-t['r0']
    states = (*[score>=i for i in range(5)],(score==4)&(distance>=2)&(distance<5))
    for a,b in RANGES:
        for lo,hi in zip(MAG[:-1],MAG[1:]):
            q = (t['r']>=lo)&(t['r']<hi)&(c>=a)&(c<b)
            counts = np.array([np.bincount(k[q&w],minlength=n) for w in states])
            if np.any(np.diff(counts,axis=0)>0):
                raise ValueError('Selection counts are not nested')
            for i in np.flatnonzero((area>0)|(counts[0]>0)):
                cells.append(dict(side=side,pixel=int(i),r_lo=lo,r_hi=hi,g_r_lo=a,g_r_hi=b,
                    area_deg2=float(area[i]),common_depth=bool(clean_area[i]),
                    **dict(zip(LEVELS,counts[:,i]))))
            for d,e in zip(LAT[:-1],LAT[1:]):
                j = clean_area&(abs(bc)>=d)&(abs(bc)<e)
                if not j.any():
                    continue
                profiles.append(dict(side=side,r_lo=lo,r_hi=hi,g_r_lo=a,g_r_hi=b,
                    b_lo=d,b_hi=e,area_deg2=float(area[j].sum()),**dict(zip(LEVELS,counts[:,j].sum(axis=1)))))


def main():
    require_compute()
    out = DEST/'selection_controls'
    out.mkdir(exist_ok=True,parents=True)
    inputs,cells,profiles = {},[],[]
    for side in ('north','south'):
        run(side,inputs,cells,profiles)
    files = (out/'sky_selection.csv',out/'latitude_selection.csv')
    for p,rows in zip(files,(cells,profiles)):
        csv_rows(p,rows)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        script=sha(ROOT/'scripts/final_science_selection.py'),outputs={p.name:sha(p) for p in files}))


if __name__=='__main__':
    main()

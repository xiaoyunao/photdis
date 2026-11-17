#!/usr/bin/env python3
"""Frozen GD-1 geometry and DR11 coverage before distance inference."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from photdis.photometry import CLEAN_REJECT_MASKBITS
from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_5_dr11_area import BANDS, random_memmap
from v3_9_catalog import CONFIG as CATALOG, settings

CONFIG = ROOT/'configs/v3_16_gd1.json'
P = json.loads(CONFIG.read_text())
DEST = ROOT/'data/v3_16'
EDGES = np.asarray(P['phi1_edges'])
OFFSETS = np.asarray(P['phi2_offsets'])
LIMITS = np.asarray(P['r_limits'])
WIDTH = P['ribbon_half_width_deg']
ROTATION = np.asarray(P['rotation'])
NSEG = len(EDGES)-1
NSTEP = int((EDGES[1]-EDGES[0])/P['sky_block_deg'])
BLOCKS = int((EDGES[-1]-EDGES[0])/P['sky_block_deg'])
MASK = sum(1<<i for i in (*CLEAN_REJECT_MASKBITS,13))


def unit(lon,lat):
    l,b = np.deg2rad([lon,lat])
    return np.array([np.cos(l)*np.cos(b),np.sin(l)*np.cos(b),np.sin(b)])


def angles(v):
    return np.rad2deg(np.arctan2(v[1],v[0])),np.rad2deg(np.arcsin(np.clip(v[2],-1,1)))


def coordinates(ra,dec):
    return angles(ROTATION@unit(ra,dec))


def track(phi1):
    return np.interp(phi1,P['track_phi1'],P['track_phi2'])


def classify(ra,dec):
    l,b = coordinates(ra,dec)
    delta = b-track(l)
    j = np.full(len(l),-1,dtype='i2')
    for i,offset in enumerate(OFFSETS):
        j[np.abs(delta-offset)<WIDTH] = i
    block = np.floor((l-EDGES[0])/P['sky_block_deg']).astype('i2')
    ok = (block>=0)&(block<BLOCKS)&(j>=0)
    return l,delta,block,j,ok


def bounding_box():
    lon = np.linspace(EDGES[0],EDGES[-1],401)
    lo,hi = min(OFFSETS)-WIDTH,max(OFFSETS)+WIDTH
    ra,dec = angles(ROTATION.T@unit(np.tile(lon,2),np.r_[track(lon)+lo,track(lon)+hi]))
    return ra.min()-1,dec.min()-1,ra.max()+1,dec.max()+1


def required_depth():
    mag = LIMITS[:,None]+np.asarray(P['footprint_grz_minus_r'])
    flux = 10**((22.5-mag)/2.5)
    snr = 2.5/np.log(10)/P['maximum_grz_error_mag']
    return (snr/flux)**2


def count_randoms(path):
    rows,density,names = random_memmap(path)
    ra0,dec0,ra1,dec1 = bounding_box()
    out = np.zeros((4,BLOCKS,len(OFFSETS)),int)
    depth = required_depth()
    release = settings()['north']['release']
    for i in range(0,len(rows),350000):
        x = rows[i:i+350000]
        ra,dec = x[names['RA']],x[names['DEC']]
        k = (ra>=ra0)&(ra<=ra1)&(dec>=max(dec0,P['north_declination_min']))&(dec<=dec1)
        x,ra,dec = x[k],ra[k],dec[k]
        _,_,b,j,ok = classify(ra,dec)
        g = galactic(ra,dec)[1]
        ok &= (g>P['minimum_galactic_latitude'])&(x[names['RELEASE']]==release)
        ok &= (x[names['EBV']]>=0)&(x[names['EBV']]<P['maximum_ebv'])
        ok &= ((x[names['MASKBITS']].astype('i8')&MASK)==0)&(np.char.strip(x[names['PHOTSYS']])==b'N')
        ok &= np.column_stack([x[names[f'NOBS_{band}']]>0 for band in BANDS]).all(axis=1)
        dp = np.column_stack([x[names[f'PSFDEPTH_{band}']] for band in BANDS])
        np.add.at(out[0],(b[ok],j[ok]),1)
        for a,threshold in enumerate(depth,1):
            use = ok&(dp>=threshold).all(axis=1)
            np.add.at(out[a],(b[use],j[use]),1)
    return out/density


def coverage():
    root = Path(settings()['north']['sweep_root']).parent.parent/'randoms'
    path = root/'randoms-north-1-0.fits'
    area = count_randoms(path)
    # Each ribbon spans 5 longitude degrees and 0.5 latitude degrees.
    nominal = np.diff(EDGES)*2*WIDTH
    sums = area.reshape((4,NSEG,NSTEP,len(OFFSETS))).sum(axis=2)
    good = sums[1:]>=P['minimum_usable_area_fraction']*nominal[None,:,None]
    passed = good[:,:,0]&(good[:,:,1:].sum(axis=2)>=P['minimum_controls'])
    records = [dict(phi1_lo=EDGES[s],phi1_hi=EDGES[s+1],phi2_offset=off,
                    clean_area_deg2=sums[0,s,j],area_r20_deg2=sums[1,s,j],
                    area_r21_deg2=sums[2,s,j],area_r22_deg2=sums[3,s,j])
               for s in range(NSEG) for j,off in enumerate(OFFSETS)]
    folder = DEST/'coverage'
    folder.mkdir(parents=True,exist_ok=True)
    csv_rows(folder/'area.csv',records)
    atomic_npz(folder/'area.npz',area=area,coverage_ok=good,passed=passed)
    st = path.stat()
    save_json(folder/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),config=sha(CONFIG),
              catalog_config=sha(CATALOG),random_path=str(path),random_size=st.st_size,
              random_mtime_ns=st.st_mtime_ns,passing_segments=passed.sum(axis=1).tolist(),
              outputs={k:sha(folder/k) for k in ('area.csv','area.npz')}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('coverage',))
    if parser.parse_args().stage=='coverage':
        coverage()

#!/usr/bin/env python3
"""Check GD-1 output hashes and audit local imaging around external members."""
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_5_dr11_area import random_memmap
from v3_10_figures import check_templates
from v3_13_delivery import bind, figure_check
from v3_16_gd1 import unit
from v3_17_sample import CONFIG, DEST, P, settings

RADIUS_DEG = .04


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def local_coverage(path,ra,dec):
    rows,_,names = random_memmap(path)
    source = unit(ra,dec).T
    radius = 2*np.sin(np.deg2rad(RADIUS_DEG)/2)
    lo,hi = np.array([ra.min(),dec.min()])-.10,np.array([ra.max(),dec.max()])+.10
    hits = np.zeros(len(ra),int)
    for first in range(0,len(rows),350000):
        q = rows[first:first+350000]
        r,b = q[names['RA']],q[names['DEC']]
        k = (r>=lo[0])&(r<=hi[0])&(b>=lo[1])&(b<=hi[1])
        k &= q[names['RELEASE']]==settings()['north']['release']
        for band in 'GRZ':
            k &= q[names['NOBS_'+band]]>0
        if k.any():
            j = np.char.strip(q[names['PHOTSYS']][k])==b'N'
            if j.any():
                points = cKDTree(unit(r[k][j],b[k][j]).T)
                hits += points.query_ball_point(source,radius,return_length=True)
    return hits


def run():
    inp = {'config':sha(CONFIG),'style':check_templates()}
    h = {}
    for stage in ('sample','members','hess'):
        h[stage] = bind(DEST/stage/'complete.json',inp)
        assert h[stage]['inputs']['config']==inp['config']
    assert h['hess']['inputs']['style']==inp['style']
    for f in (DEST/'hess').glob('*.png'):
        figure_check(f)
    rec = read(DEST/'members'/'members.csv')
    n = len(rec)
    assert h['members']['rows']==n and n==len({(r['catalog'],r['source_id']) for r in rec})
    for x in rec:
        if x['clean']=='True' or x['qrf_archived']=='True':
            assert x['native_type']=='PSF'
        if x['legacy_qrf']=='True' or x['desi_qrf']=='True':
            assert x['qrf_archived']=='True'
    assert h['sample']['populations']['legacy']==[175328,142098]
    random = Path(settings()['north']['sweep_root']).parent.parent/'randoms/randoms-north-1-0.fits'
    st = random.stat()
    inp['random'] = dict(path=str(random),size=st.st_size,mtime_ns=st.st_mtime_ns)
    ra,dec = [np.array([float(r[k]) for r in rec]) for k in ('ra','dec')]
    exposure = local_coverage(random,ra,dec)
    groups = []
    for src,group in sorted({(r['catalog'],r['group']) for r in rec}):
        j = [i for i,r in enumerate(rec) if (r['catalog'],r['group'])==(src,group)]
        rows = [rec[i] for i in j]
        available = (dec[j]>=P['north_declination_min'])&(exposure[j]>0)
        count = [r['match'] in ('gaia_id','unique_position') for r in rows]
        groups.append(dict(catalog=src,group=group,n=len(j),near_north_imaging=int(available.sum()),
            matched=int(sum(count)),matched_near_north_imaging=int((available&count).sum()),
            psf=sum(r['psf']=='True' for r in rows),clean=sum(r['clean']=='True' for r in rows),
            qrf_archived=sum(r['qrf_archived']=='True' for r in rows)))
    out = DEST/'delivery'
    out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'imaging.csv',[dict(catalog=x['catalog'],source_id=x['source_id'],
        north_randoms_with_grz_within_0p04deg=int(n),north_declination=bool(d>=P['north_declination_min']))
                               for x,n,d in zip(rec,exposure,dec)])
    csv_rows(out/'member_summary.csv',groups)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inp,
        distance_gate=h['hess']['authorize_distance_stage'],signal_segments=h['hess']['signal_segments'],
        outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()

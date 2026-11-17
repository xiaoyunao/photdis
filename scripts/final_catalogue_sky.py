#!/usr/bin/env python3
"""Accumulate full-release sky and space statistics from each native PDF partition."""
import json
import os

import healpy as hp
import numpy as np

from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_unified import owner_rows
from final_science_catalog import indices, REGIONS
from final_science_evidence import SHELLS
from final_catalogue_raw import DEST

NSIDE=16
NPIX=hp.nside2npix(NSIDE)
NX,NY=1440,720
SHARDS=32
R=(-np.inf,16.,18.,20.,21.,22.,23.,np.inf)
C=(-np.inf,.2,.5,.8,1.1,1.4,np.inf)
XY=np.linspace(-10.,10.,241)
LEVELS=('native_parent','full_pdf','science_core')
PLANES=((0,1),(0,2),(1,2))


def projected(l,b):
    u=-np.deg2rad((l+180)%360-180)
    v=np.deg2rad(b)
    co=np.cos(v)
    a=np.arccos(np.clip(co*np.cos(u/2),-1.,1.))
    den=np.sinc(a/np.pi)
    xp=2*co*np.sin(u/2)/den
    yp=np.sin(v)/den
    x=np.clip(((xp/np.pi+1)*NX/2).astype(int),0,NX-1)
    y=np.clip(((yp/(np.pi/2)+1)*NY/2).astype(int),0,NY-1)
    return x+y*NX,u,v


def selected_ids():
    folder=ROOT/'data/final_science/evidence'
    mark=folder/'complete.json'
    p=verified(folder/'selected.npz',json.loads(mark.read_text())['outputs']['selected.npz'])
    t=read_columns(p,('ls_id','TYPE','science'))
    assert (t['TYPE']=='PSF').all()
    return np.sort(t['ls_id'][t['science']<=2]),sha(mark)


def one(part,moves,science,hist,area,images,xyz,soft,dist):
    region=REGIONS.index(part['region'])
    frozen=part['region']=='south_v35'
    cols=('ls_id','TYPE','RA','DEC')
    v=('observed_r','dereddened_g_r','qrf_quantiles_kpc','qrf_bins') if frozen else (
       'r','g0','r0','north_primary_region')
    p=read_columns(verified(ROOT/part['photo_file'],part['photo_sha256']),cols+v)
    own,_=owner_rows(part['side'],p,moves,frozen=frozen)
    assert len(own)==int(part['native_rows']) and own.sum()==int(part['owned_rows'])
    if frozen:
        i=np.flatnonzero(own)
        q=p['qrf_quantiles_kpc'][i,2]
        bins=p['qrf_bins'][i]
    elif part['pdf_file']:
        raw=read_columns(verified(ROOT/part['pdf_file'],part['pdf_sha256']),
              ('ls_id','TYPE','photo_row','qrf_quantiles_kpc','qrf_bins'))
        k=raw['photo_row']
        np.testing.assert_array_equal(raw['ls_id'],p['ls_id'][k])
        np.testing.assert_array_equal(raw['TYPE'],p['TYPE'][k])
        w=own[k]
        i=k[w]
        q=raw['qrf_quantiles_kpc'][w,2]
        bins=raw['qrf_bins'][w]
    else:
        i=np.empty(0,'i8')
        q=np.empty(0)
        bins=np.empty((0,len(SHELLS)-1))
    assert len(i)==int(part['owned_pdf_rows'])
    if len(i):
        np.testing.assert_allclose(bins.sum(axis=1),1,rtol=0,atol=1e-7)
    ids=p['ls_id'][i]
    j=np.searchsorted(science,ids)
    z=j<len(science)
    z[z]=science[j[z]]==ids[z]
    near=i[z]
    r=p['observed_r'] if frozen else p['r']
    c=p['dereddened_g_r'] if frozen else p['g0']-p['r0']
    ri=np.searchsorted(R,r,side='right')-1
    ci=np.searchsorted(C,c,side='right')-1
    ri=np.where(np.isfinite(r),np.clip(ri,0,len(R)-2),len(R)-1)
    ci=np.where(np.isfinite(c),np.clip(ci,0,len(C)-2),len(C)-1)
    # All three source views use actual coordinates and shared projected pixels.
    jall=np.flatnonzero(own)
    l,b=galactic(p['RA'][jall],p['DEC'][jall])
    pixel,ul,vl=projected(l,b)
    hpix=hp.ang2pix(NSIDE,p['RA'][jall],p['DEC'][jall],nest=True,lonlat=True)
    for k,sel in enumerate((jall,i,near)):
        j=np.searchsorted(jall,sel)
        assert np.array_equal(jall[j],sel)
        ix=(ri[sel]*len(C)+ci[sel])*NPIX+hpix[j]
        u,n=np.unique(ix,return_counts=True)
        hist[region,k].reshape(-1)[u]+=n
        u,n=np.unique(pixel[j],return_counts=True)
        images[k].reshape(-1)[u]+=n
        area[region,k]+=np.bincount(hpix[j],minlength=NPIX)
        if not k:
            continue
        d,m=(q,bins) if k==1 else (q[z],bins[z])
        d50=np.searchsorted(SHELLS,d,side='right')-1
        assert np.all((d50>=0)&(d50<len(SHELLS)-1))
        dist[region,k-1,0]+=np.bincount(d50,minlength=len(SHELLS)-1)
        dist[region,k-1,1]+=m.sum(axis=0)
        for h in range(len(SHELLS)-1):
            soft[region,k-1,0,h]+=np.bincount(hpix[j],d50==h,minlength=NPIX)
            soft[region,k-1,1,h]+=np.bincount(hpix[j],m[:,h],minlength=NPIX)
        if k==1:
            for h,(lo,hi) in enumerate(((0,1),(1,5),(5,10),(0,10))):
                u,n=np.unique(pixel[j[(d>=lo)&(d<hi)]],return_counts=True)
                images[h+3].reshape(-1)[u]+=n
        take=d<10
        d=d[take]
        ll,bb=-ul[j[take]],vl[j[take]]
        xy=(d*np.cos(bb)*np.cos(ll),d*np.cos(bb)*np.sin(ll),d*np.sin(bb))
        for h,(a,b) in enumerate(PLANES):
            counts=np.histogram2d(xy[a],xy[b],bins=(XY,XY))[0]
            assert counts.sum()==len(d)
            xyz[k-1,h]+=counts.astype('i8')


def main():
    require_compute()
    k=int(os.environ['SLURM_ARRAY_TASK_ID'])
    parts,moves,inputs=indices()
    science,inputs['science']=selected_ids()
    npix=NPIX
    hist=np.zeros((4,3,len(R),len(C),npix),'i8')
    area=np.zeros((4,3,npix),'i8')
    images=np.zeros((7,NY,NX),'i8')
    xyz=np.zeros((2,3,len(XY)-1,len(XY)-1),'i8')
    n=len(SHELLS)-1
    soft=np.zeros((4,2,2,n,npix))
    dist=np.zeros((4,2,2,n))
    where=np.array_split(np.arange(len(parts)),SHARDS)[k]
    for j in where:
        one(parts[j],moves,science,hist,area,images,xyz,soft,dist)
    np.testing.assert_array_equal(hist.sum(axis=(2,3)),area)
    out=DEST/'sky'
    out.mkdir(exist_ok=True,parents=True)
    p=out/f'{k:02}.npz'
    atomic_npz(p,hist=hist,counts_by_pixel=area,images=images,xyz=xyz,soft=soft,dist=dist)
    save_json(p.with_suffix('.json'),dict(job=os.getenv('SLURM_JOB_ID'),parts=len(where),inputs=inputs,
         outputs={p.name:sha(p)}))
    print(k,'sources',area.sum(axis=(0,2)),flush=True)


if __name__=='__main__':
    main()

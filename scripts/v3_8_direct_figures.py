#!/usr/bin/env python3
"""Plot direct validation and individual-coordinate catalog views."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from photdis.stage1 import galactic, require_compute
from photdis.v3.metrics import distance_summary
from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_catalog_projection import OUT as PROJECTION
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_reference_photometry import OUT as PHOTO
from v3_6_vac_distances import read_columns
from v3_7_reference_comparison import SOURCES, load_source
from v3_7_selection_replay import OUT as SELECTION, probabilities
from v3_8_candidate import gates, read_cache

OUT = ROOT/'data/v3_8/direct_figures'
SHELLS = ((0.,1.),(1.,5.),(5.,10.),(0.,10.))
XCOL = np.linspace(-.6,2.9,141)
RCOL = np.linspace(-.6,2.4,121)
MABS = np.linspace(-8.,20.,141)
MAPP = np.linspace(12.,20.2,124)
DIST = np.geomspace(.008,80,120)
RES = np.linspace(-5.,5.,121)
SPACE = np.linspace(-10,10,241)
COMMON = ('catalog_row','ls_id','TYPE','RA','DEC','reference_kpc','high_quality','qrf_training',
          'recorded_history','consensus_class','class_disagreement','qrf_quantiles_kpc','primary')
CATALOG = ('catalog_row','ls_id','TYPE','RA','DEC','primary','nominal_qnear','observed_r',
           'dereddened_g_r','dwarf_selection_score','band_state','qrf_quantiles_kpc')


def save(fig,dest,name,rows,data=None):
    for suffix in ('pdf','png'):
        fig.savefig(dest/f'{name}.{suffix}',dpi=300)
    plt.close(fig)
    csv_rows(dest/f'{name}.csv',rows)
    if data:
        atomic_npz(dest/f'{name}.npz',**data)


def density(fig,ax,x,y,xb,yb,label='Stars per cell'):
    hist = np.histogram2d(x,y,bins=(xb,yb))[0]
    actual = np.isfinite(x)&np.isfinite(y)
    if 0<actual.sum()<=300:
        ax.scatter(x[actual],y[actual],s=9,lw=0,color=BLUE,alpha=.65,rasterized=True)
    elif hist.max()>0:
        img = ax.pcolormesh(xb,yb,np.ma.masked_equal(hist.T,0),norm=LogNorm(1,max(2,hist.max())),
                            cmap='viridis',rasterized=True)
        fig.colorbar(img,ax=ax,pad=.02,label=label)
    ax.set(xlim=(xb[0],xb[-1]),ylim=(yb[0],yb[-1]))
    return hist.astype('i8')


def reference_figures(dest):
    marker = EVIDENCE/'complete.json'
    evidence = json.loads(marker.read_text())
    photo = read_columns(read_cache(PHOTO,'photometry.npz'),('catalog_row','ls_id','TYPE','magnitude'))
    if np.any(np.diff(photo['catalog_row'])<=0):
        raise ValueError('Reference photometry catalog keys must be unique and sorted')
    f,axs = plt.subplots(2,3,figsize=(18,9),layout='constrained',sharex='col')
    c,axes = plt.subplots(3,2,figsize=(13,15),layout='constrained',sharex=True,sharey=True)
    distance_tables,cmd_tables,counts,cmd_counts = {},{},{},[]
    excluded = {}
    for n,source in enumerate(SOURCES):
        a,excluded[source] = load_source(source,evidence)
        j = np.flatnonzero(a['primary'])
        k = np.searchsorted(photo['catalog_row'],a['catalog_row'][j])
        np.testing.assert_array_equal(photo['ls_id'][k],a['ls_id'][j])
        if not (photo['TYPE'][k]=='PSF').all():
            raise ValueError('Reference CMD needs verified native PSFs')
        d,q = a['reference_kpc'][j],a['qrf_quantiles_kpc'][j]
        mag = photo['magnitude'][k]
        residual = 5*np.log10(q[:,2]/d)
        xgr = mag[:,0]-mag[:,1]
        ax,bx = axs[:,n]
        dh = density(f,ax,d,q[:,2],DIST,DIST)
        rh = density(f,bx,d,residual,DIST,RES)
        ax.plot(DIST[[0,-1]],DIST[[0,-1]],ls='--',color=ORANGE,lw=1.5)
        bx.axhline(0,ls='--',color=ORANGE,lw=1.25)
        ax.set(xscale='log',yscale='log',title=f'{source}   N = {len(j):,}')
        bx.set(xscale='log',xlabel='Reference distance (kpc)')
        row = distance_summary(10+5*np.log10(q),10+5*np.log10(d))
        counts[source] = dict(source=source,**row,distance_exterior=int(len(j)-dh.sum()),
                              residual_exterior=int(len(j)-rh.sum()))
        ax.text(.05,.94,f"Bias {100*row['median_relative']:+.1f}%   ·   NMAD {100*row['nmad_relative']:.1f}%",
                transform=ax.transAxes,va='top',fontsize=11,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.83})
        bx.text(.05,.06,f'{len(j)-rh.sum():,} outside axes',transform=bx.transAxes,fontsize=10,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.83})
        distance_tables.update({source+'_distance':dh,source+'_residual':rh})
        keep = np.isfinite(xgr)&np.isfinite(mag[:,1])
        for i,(z,title) in enumerate(((d,'Reference distance'),(q[:,2],'Photometric distance'))):
            ay = axes[n,i]
            hh = density(c,ay,xgr[keep],mag[keep,1]-10-5*np.log10(z[keep]),XCOL,MABS)
            ay.set(ylim=(MABS[-1],MABS[0]),title=f'{source} · {title} · N = {keep.sum():,}')
            cmd_tables[source+str(i)] = hh
            cmd_counts.append(dict(source=source,distance=title,rows=int(keep.sum()),outside=int(keep.sum()-hh.sum())))
    axs[0,0].set_ylabel('Photometric median distance (kpc)')
    axs[1,0].set_ylabel('Distance modulus residual (mag)')
    f.suptitle('Distance validation · primary selection and qualified references outside QRF training')
    axes[-1,0].set_xlabel('Dereddened g − r (mag)')
    axes[-1,1].set_xlabel('Dereddened g − r (mag)')
    for ax in axes[:,0]:
        ax.set_ylabel('Absolute r magnitude')
    c.suptitle('Reference and photometric CMDs for the same stars')
    save(f,dest,'F02_direct_distances',counts.values(),dict(distance_bins=DIST,residual_bins=RES,**distance_tables))
    save(c,dest,'F03_same_star_cmd',cmd_counts,dict(color_bins=XCOL,absolute_bins=MABS,**cmd_tables))
    return excluded


def catalog():
    meta = json.loads((PROJECTION/'complete.json').read_text())
    catalog_parts = []
    for key,digest in meta['child_receipts'].items():
        marker = verified(ROOT/key,digest)
        catalog_parts.append(read_columns(read_cache(marker.parent,'rows.npz'),CATALOG))
    raw = {k:np.concatenate([x[k] for x in catalog_parts]) for k in CATALOG}
    sort = np.argsort(raw['catalog_row'])
    if not np.array_equal(raw['catalog_row'][sort],np.arange(len(sort))):
        raise ValueError('Full catalog keys must have exact coverage')
    if not (raw['TYPE']=='PSF').all():
        raise ValueError('Full catalog is a verified native PSF-only parent')
    return {k:v[sort] for k,v in raw.items() if k!='TYPE'}


def matched_roster(cat):
    n = len(cat['ls_id'])
    d = np.full(n,np.nan)
    label = np.full(n,-1,'i1')
    role = np.zeros(n,'u1')
    conflict = np.zeros(n,bool)
    known = np.zeros(n,bool)
    meta = json.loads((EVIDENCE/'complete.json').read_text())
    for source in ('Gaia','SpecDis','SPDist',*(x for x in meta['rows'] if x not in SOURCES)):
        a = read_columns(read_cache(EVIDENCE,f'{source}.npz'),COMMON)
        k = a['high_quality']&np.isfinite(a['reference_kpc'])&(a['reference_kpc']>0)
        j = a['catalog_row'][k]
        np.testing.assert_array_equal(a['ls_id'][k],cat['ls_id'][j])
        if not np.all(a['TYPE'][k]=='PSF') or len(np.unique(j))!=len(j):
            raise ValueError('One qualified native PSF per source and catalog row required')
        first = ~known[j]
        d[j[first]] = a['reference_kpc'][k][first]
        role[j] |= a['qrf_training'][k].astype('u1') | (a['recorded_history'][k].astype('u1')<<1) | 4
        cls = a['consensus_class'][k]
        use = cls>=0
        conflict[j] |= a['class_disagreement'][k]|((label[j]!=cls)&(label[j]>=0)&use)
        label[j[use]] = cls[use]
        known[j] = True
    label[conflict] = -1
    return d,label,known,role


def scatter_sky(ax,glon,glat,color,s=.065,alpha=.26):
    ax.scatter(glon,glat,s=s,c=color,alpha=alpha,linewidths=0,rasterized=True)
    ax.set_xticklabels([])
    ax.set_yticks(np.deg2rad([-60,-30,0,30,60]))


def aitoff(dest,name,title,distance,l,b,mask,footprint,color):
    fig,axes = plt.subplots(2,2,subplot_kw={'projection':'aitoff'},figsize=(22,12),layout='constrained')
    counts = []
    for ax,(low,high) in zip(axes.ravel(),SHELLS):
        select = mask&(distance>=low)&(distance<high)
        ax.contourf(*footprint[:2],footprint[2],levels=[.5,1.5],colors=['#eeeeee'],alpha=.85)
        scatter_sky(ax,l[select],b[select],color)
        ax.set_title(f'{low:g}–{high:g} kpc   ·   {select.sum():,} sources',pad=10)
        counts.append(dict(lo_kpc=low,hi_kpc=high,n=int(select.sum()),view=name))
    if sum(r['n'] for r in counts[:3])!=counts[3]['n']:
        raise ValueError('Sky shell union mismatch')
    fig.suptitle(title+'\nGalactic coordinates · gray: sampled catalog footprint · individual source positions',fontsize=21)
    save(fig,dest,name,counts)


def spatial(dest,cat,d,l,b,ref,has_ref):
    f,axes = plt.subplots(2,3,figsize=(17,10),layout='constrained')
    rows,data = [],{}
    for row,(name,z,sel) in enumerate((('Primary photometric distances',d,cat['primary']),
                                      ('Matched qualified reference distances',ref,cat['primary']&has_ref))):
        mask = sel&(z>0)&(z<10)
        dd,ll,bb = z[mask],-l[mask],b[mask]
        xyz = (dd*np.cos(bb)*np.cos(ll),dd*np.cos(bb)*np.sin(ll),dd*np.sin(bb))
        for j,(a,b_,x,y) in enumerate(((0,1,'X','Y'),(0,2,'X','Z'),(1,2,'Y','Z'))):
            ax = axes[row,j]
            h = density(f,ax,xyz[a],xyz[b_],SPACE,SPACE,'Stars per projected cell')
            np.testing.assert_equal(h.sum(),mask.sum())
            ax.set(xlabel=f'{x} (kpc)',ylabel=f'{y} (kpc)',title=f'{x}–{y} · N = {mask.sum():,}')
            ax.set_aspect('equal')
            rows.append(dict(sample=name,projection=x+y,n=int(mask.sum())))
            data[f'hist_{row}_{j}'] = h
    f.suptitle('Heliocentric projections · upper: photometric distances · lower: qualified references\n'
               'Counts within the survey footprint, 0–10 kpc')
    save(f,dest,'F05_heliocentric_projection',rows,dict(axes=SPACE,**data))


def overview(dest,cat,mask,footprint,l,b):
    f = plt.figure(figsize=(17,10),layout='constrained')
    grid = f.add_gridspec(2,2)
    sky = f.add_subplot(grid[0,:],projection='aitoff')
    sky.contourf(*footprint[:2],footprint[2],levels=[.5,1.5],colors=['#eeeeee'])
    scatter_sky(sky,l,b,BLUE,s=.04,alpha=.18)
    scatter_sky(sky,l[mask],b[mask],ORANGE,s=.06,alpha=.23)
    sky.set_title(f'Galactic sky · {len(b):,} modeled sources · {mask.sum():,} qualified reference matches')
    ax,cx = f.add_subplot(grid[1,0]),f.add_subplot(grid[1,1])
    h = density(f,ax,cat['dereddened_g_r'],cat['observed_r'],XCOL,MAPP)
    ax.set(xlabel='Dereddened g − r (mag)',ylabel='Observed r (mag)',ylim=(20.2,12.),title='Catalog color and magnitude')
    photo = read_columns(read_cache(PHOTO,'photometry.npz'),('TYPE','magnitude','catalog_row'))
    use = mask[photo['catalog_row']]
    if not (photo['TYPE'][use]=='PSF').all():
        raise ValueError('Color comparison requires native PSFs')
    g,r,z = photo['magnitude'][use][:,[0,1,3]].T
    hh = density(f,cx,g-r,r-z,XCOL,RCOL)
    cx.set(xlabel='Dereddened g − r (mag)',ylabel='Dereddened r − z (mag)',title='Qualified reference color–color plane')
    save(f,dest,'F01_catalog_overview',
         [dict(panel='catalog',rows=len(b),in_axes=int(h.sum())),dict(panel='matched',rows=int(mask.sum()),in_axes=int(hh.sum()))],
         dict(gr=XCOL,app=MAPP,rz=RCOL,observed=h,colors=hh))


def sky_figures(dest,mode):
    cat = catalog()
    ref,cls,known,role = matched_roster(cat)
    d = cat['qrf_quantiles_kpc'][:,2]
    glon,glat = galactic(cat['RA'],cat['DEC'])
    l = -np.deg2rad((glon+180)%360-180)
    b = np.deg2rad(glat)
    lx,by = np.linspace(-np.pi,np.pi,145),np.linspace(-np.pi/2,np.pi/2,73)
    area = np.histogram2d(l,b,bins=(lx,by))[0].T>0
    footprint = (.5*(lx[1:]+lx[:-1]),.5*(by[1:]+by[:-1]),area)
    if mode=='overview':
        overview(dest,cat,known,footprint,l,b)
        spatial(dest,cat,d,l,b,ref,known)
    else:
        a = gates()
        q = probabilities(cat,a)
        candidate = (cat['dwarf_selection_score']>=.65)&(q>=.90)
        variants = (
            ('catalog','Modeled catalog · QRF median',d,np.ones(len(d),bool),BLUE),
            ('primary','Primary selection · QRF median',d,cat['primary'],BLUE),
            ('candidate_known','Candidate verified subset · QRF median',d,candidate,BLUE),
            ('references','Unique qualified references · reference distance',ref,known,ORANGE),
            ('dwarfs_reference','Reliable dwarf labels · reference distance',ref,known&(cls==0),GREEN),
            ('giants_reference','Reliable giant labels · reference distance',ref,known&(cls==1),ORANGE),
            ('dwarfs_qrf','Same reliable dwarf labels · QRF median',d,known&(cls==0),GREEN),
            ('giants_qrf','Same reliable giant labels · QRF median',d,known&(cls==1),ORANGE))
        for name,title,z,mask,color in variants:
            aitoff(dest,'F04_'+name,title,z,l,b,mask,footprint,color)
        csv_rows(dest/'reference_roles.csv',
                 (dict(qrf_training=bool(i&1),recorded_history=bool(i&2),qualified_reference=bool(i&4),
                       rows=int((role==i).sum())) for i in np.unique(role)))


def main():
    require_compute()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('reference','overview','sky'))
    args = p.parse_args()
    dest = OUT/args.mode
    dest.mkdir(parents=True,exist_ok=True)
    marker = dest/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    with plt.rc_context(STYLE):
        exclusions = reference_figures(dest) if args.mode=='reference' else None
        if args.mode!='reference':
            sky_figures(dest,args.mode)
    files = [EVIDENCE/'complete.json',PROJECTION/'complete.json',PHOTO/'complete.json',Path(__file__).resolve()]
    if args.mode=='sky':
        files.append(SELECTION/'complete.json')
    save_json(marker,dict(job=os.getenv('SLURM_JOB_ID'),excluded_morphology=exclusions or
        dict(non_psf=0,unverified_type=0),inputs={str(x.relative_to(ROOT)):sha(x) for x in files},
        outputs={x.name:sha(x) for x in dest.iterdir() if x.is_file() and x!=marker}))


if __name__=='__main__':
    main()

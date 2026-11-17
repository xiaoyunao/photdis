#!/usr/bin/env python3
"""Render full-release products with footprint supplied by independent randoms."""
import csv
import json
import os

import healpy as hp
from matplotlib.colors import ColorConverter, LogNorm
import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import BLUE, GREEN, ORANGE, STYLE, plt, save, scatter_sky
from v3_10_figures import check_templates
from v3_9_catalog import LEDGER
from final_science_catalog import REGIONS
from final_science_evidence import SHELLS
from final_catalogue_raw import DEST, GROUPS
from final_catalogue_sky import LEVELS, NSIDE, R, C, XY, SHARDS

OUT=DEST/'atlas'
SKY_FIELDS=('hist','counts_by_pixel','images','xyz','soft','dist')


def reduce_sources(inputs):
    agg={}
    n=0
    for i in range(SHARDS):
        p=DEST/'sky'/f'{i:02}.json'
        r=json.loads(p.read_text())
        inputs[str(p.relative_to(ROOT))]=sha(p)
        n+=r['parts']
        x=read_columns(verified(p.with_suffix('.npz'),r['outputs'][p.with_suffix('.npz').name]),SKY_FIELDS)
        for k,v in x.items():
            if k not in agg:
                agg[k]=v
            else:
                agg[k]+=v
    area=agg['counts_by_pixel']
    assert n==28917 and tuple(area.sum(axis=(0,2)))==(72289386,43347612,2281023)
    np.testing.assert_array_equal(agg['images'][:3].sum(axis=(1,2)),area.sum(axis=(0,2)))
    np.testing.assert_equal(agg['images'][3:].sum(axis=(1,2))[:3].sum(),agg['images'][6].sum())
    np.testing.assert_array_equal(agg['hist'].sum(axis=(2,3)),area)
    np.testing.assert_allclose(agg['dist'].sum(axis=-1),area[:,1:].sum(axis=-1)[:,:,None]
                               *np.ones((1,1,2)),rtol=0,atol=2e-6)
    np.testing.assert_allclose(agg['dist'],agg['soft'].sum(axis=-1),rtol=0,atol=2e-6)
    return agg


def random_areas(inputs):
    regions={}
    for s in ('north','south'):
        p=ROOT/'data/final_science/area'/s/'complete.json'
        z=json.loads(p.read_text())
        inputs[str(p.relative_to(ROOT))]=sha(p)
        data=read_columns(verified(p.with_name('area.npz'),z['outputs']['area.npz']),
                          ('effective_deg2',))['effective_deg2']
        assert z['nside']==NSIDE
        if s=='north':
            regions['north']=data[1]
        else:
            regions.update(south_v35=data[1],south_faint=data[1],south_low=data[0])
    return np.stack([regions[k] for k in REGIONS])


def sky_grid(random):
    l=np.linspace(-np.pi,np.pi,361)
    b=np.linspace(-np.pi/2,np.pi/2,181)
    ll,bb=np.meshgrid(l,b)
    ra,dec=hp.Rotator(coord=('G','C'))(-np.rad2deg(ll.ravel()),np.rad2deg(bb.ravel()),lonlat=True)
    pix=hp.ang2pix(NSIDE,ra,dec,lonlat=True,nest=True).reshape(ll.shape)
    return l,b,random[pix]>0,pix


def backdrop(ax,grid):
    ax.contourf(*grid[:3],levels=[.5,1.5],colors=['#eeeeee'],alpha=.85)
    ax.set_xticklabels([])
    ax.set_yticks(np.deg2rad([-60,-30,0,30,60]))


def source_plane(ax,a,level):
    data=np.empty((*a.shape,4),dtype='f4')
    data[:,:,:3]=ColorConverter.to_rgb(BLUE)
    data[:,:,3]=np.where(a>0,np.clip(.18+.82*np.log1p(a)/np.log1p(level),0,1),0)
    # The image coordinates are projected Aitoff pixels, never bin-centre markers.
    ax.imshow(data,origin='lower',extent=(0,1,0,1),transform=ax.transAxes,
              interpolation='nearest',clip_path=ax.patch,zorder=2)


def sky_figures(a,random):
    full=random[[0,2,3]].sum(axis=0)
    grid=sky_grid(full[0])
    images=a['images']
    cap=np.quantile(images[0][images[0]>0],.98)
    fig,axes=plt.subplots(2,2,subplot_kw={'projection':'aitoff'},figsize=(22,12),layout='constrained')
    rows=[]
    for i,(ax,label) in enumerate(zip(axes.ravel(),('Owned native PSFs','Complete distance PDFs',
                                                        'Core science sources','Complete PDFs per effective deg²'))):
        backdrop(ax,grid)
        if i<3:
            source_plane(ax,images[i],cap)
            n=int(images[i].sum())
            ax.set_title(f'{label} · {n:,}')
            rows.append(dict(view=label,n=n))
        else:
            den=full[1]
            num=a['counts_by_pixel'][:,1].sum(axis=0)
            rate=np.divide(num,den,out=np.full_like(den,np.nan),where=den>0)
            v=rate[grid[3]]
            mesh=ax.pcolormesh(*grid[:2],np.ma.masked_invalid(v),norm=LogNorm(1,np.nanpercentile(rate,98)),
                               cmap='viridis',shading='auto',rasterized=True)
            fig.colorbar(mesh,ax=ax,pad=.025,shrink=.72,label='PDF sources per deg²')
            ax.set_title(label)
            rows.append(dict(view=label,n=int(num[den>0].sum())))
    fig.suptitle('Photdis full release · individual-source raster · gray: DR11 random-point footprint',fontsize=21)
    save(fig,OUT,'F13_catalogue_sky',rows)

    fig,axes=plt.subplots(2,2,subplot_kw={'projection':'aitoff'},figsize=(22,12),layout='constrained')
    rows=[]
    for ax,im,(lo,hi) in zip(axes.ravel(),images[3:],((0,1),(1,5),(5,10),(0,10))):
        backdrop(ax,grid)
        source_plane(ax,im,cap)
        n=int(im.sum())
        ax.set_title(f'{lo:g}–{hi:g} kpc · {n:,} sources',pad=10)
        rows.append(dict(lo=lo,hi=hi,n=n))
    fig.suptitle('Complete distance PDFs by median distance · gray: DR11 random-point footprint',fontsize=21)
    save(fig,OUT,'F14_all_pdf_shells',rows)
    return grid


def selected_rerender(grid,inputs):
    mark=ROOT/'data/final_science/evidence/complete.json'
    src=verified(mark.with_name('selected.npz'),json.loads(mark.read_text())['outputs']['selected.npz'])
    t=read_columns(src,('TYPE','q','science','lon','lat','gaia_state'))
    assert (t['TYPE']=='PSF').all()
    inputs[str(mark.relative_to(ROOT))]=sha(mark)
    l=-np.deg2rad((t['lon']+180)%360-180)
    b=np.deg2rad(t['lat'])
    for name,title,cut in (
        ('F04_science_random_coverage','Selected Photdis sources · 5–10 kpc is a research tier',t['science']<=3),
        ('F04_weak_gaia_random_coverage','Gaia counterpart below geometric-distance cut · research tier included',
         (t['science']<=3)&np.isin(t['gaia_state'],(3,4,5)))):
        fig,axes=plt.subplots(2,2,subplot_kw={'projection':'aitoff'},figsize=(22,12),layout='constrained')
        rows=[]
        d=t['q'][:,2]
        for ax,(lo,hi) in zip(axes.ravel(),((0,1),(1,5),(5,10),(0,10))):
            j=cut&(d>=lo)&(d<hi)
            backdrop(ax,grid)
            scatter_sky(ax,l[j],b[j],BLUE)
            n=int(j.sum())
            ax.set_title(f'{lo:g}–{hi:g} kpc   ·   {n:,} sources',pad=10)
            rows.append(dict(lo_kpc=lo,hi_kpc=hi,n=n,view=name))
        assert sum(row['n'] for row in rows[:3])==rows[-1]['n']
        fig.suptitle(title+'\nGalactic coordinates · gray: DR11 random-point footprint · individual source positions',
                     fontsize=21)
        save(fig,OUT,name,rows)


def xyz_figure(a):
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    rows=[]
    for r,(stage,xx) in enumerate(zip(('Complete PDFs','Core science'),a['xyz'])):
        for j,(ax,mat,xy) in enumerate(zip(axes[r],xx,('XY','XZ','YZ'))):
            k=int(mat.sum())
            image=ax.pcolormesh(XY,XY,np.ma.masked_equal(mat.T,0),norm=LogNorm(1,max(2,mat.max())),
                                cmap='viridis',rasterized=True)
            fig.colorbar(image,ax=ax,pad=.02,label='Sources per projected cell')
            ax.set(xlabel=f'{xy[0]} (kpc)',ylabel=f'{xy[1]} (kpc)',title=f'{xy[0]}–{xy[1]} · {k:,} sources')
            ax.set_aspect('equal')
            rows.append(dict(stage=stage,projection=xy,n=k))
    fig.suptitle('Heliocentric projections · upper: complete PDFs · lower: core science\nMedian distances within 10 kpc')
    save(fig,OUT,'F16_catalogue_xyz',rows,dict(axes=XY,projection_counts=a['xyz']))


def strata_figure(a,random):
    hist=a['hist']
    fig,axes=plt.subplots(2,3,figsize=(18,10),layout='constrained',subplot_kw={'projection':'aitoff'})
    rows=[]
    groups=(('r < 20',(slice(0,3),slice(None))),('20 ≤ r < 21',(slice(3,4),slice(None))),
            ('21 ≤ r < 23',(slice(4,6),slice(None))),('g − r < 0.5',(slice(None),slice(0,2))),
            ('0.5 ≤ g − r < 0.8',(slice(None),slice(2,3))),('0.8 ≤ g − r',(slice(None),slice(3,6))))
    low=random[[0,2,3]].sum(axis=0)
    grid=sky_grid(low[0])
    for ax,(name,(mag,color)) in zip(axes.ravel(),groups):
        v=hist[:,1,mag,color,:].sum(axis=(1,2))
        counts=v.sum(axis=0)
        plot=np.divide(counts,low[1],where=low[1]>0,out=np.full_like(low[1],np.nan))
        backdrop(ax,grid)
        single=ax.pcolormesh(*grid[:2],np.ma.masked_invalid(plot[grid[3]]),norm=LogNorm(1,np.nanpercentile(plot,98)),
                             cmap='viridis',shading='auto',rasterized=True)
        fig.colorbar(single,ax=ax,pad=.025,shrink=.78,label='PDF sources per deg²')
        ax.set_title(f'{name} · {counts.sum():,}')
        rows.extend(dict(group=name,region=r,n=int(x.sum())) for r,x in zip(REGIONS,v))
    fig.suptitle('Complete PDFs in observed magnitude and native dereddened color strata')
    save(fig,OUT,'F17_catalogue_strata',rows)



def native_systems(a,random):
    fig,axes=plt.subplots(2,2,figsize=(22,12),layout='constrained',subplot_kw={'projection':'aitoff'})
    rows=[]
    sides=(('DECam',(0,1,2),random[[0,2]].sum(axis=0)),('BASS/MzLS',(3,),random[3]))
    plots=[]
    for j,(system,regions,area) in enumerate(sides):
        g=sky_grid(area[0])
        for i,label in enumerate(('Owned native PSFs','Complete distance PDFs')):
            count=a['counts_by_pixel'][list(regions),i].sum(axis=0)
            rate=np.divide(count,area[1],out=np.full_like(area[1],np.nan),where=area[1]>0)
            plots.append((axes[i,j],system,label,g,count,rate))
    vmax=np.nanpercentile(np.concatenate([p[-1] for p in plots]),99)
    for ax,system,label,g,count,rate in plots:
        backdrop(ax,g)
        img=ax.pcolormesh(*g[:2],np.ma.masked_invalid(rate[g[3]]),norm=LogNorm(1,vmax),
                          cmap='viridis',shading='auto',rasterized=True)
        fig.colorbar(img,ax=ax,pad=.025,shrink=.72,label='Sources per deg²')
        n=int(count.sum())
        ax.set_title(f'{system} · {label} · {n:,}')
        rows.append(dict(system=system,sample=label,n=n,n_with_area=int(count[np.isfinite(rate)].sum())))
    fig.suptitle('Native-system density · shared scale · effective area from DR11 random points',fontsize=21)
    save(fig,OUT,'F19_native_systems',rows)


def raw_flow(inputs):
    rows=[]
    for side in ('south','north'):
        for i in range(GROUPS):
            marker=DEST/'raw'/side/f'{i:02}.json'
            info=json.loads(marker.read_text())
            f=marker.with_suffix('.csv')
            inputs[str(marker.relative_to(ROOT))]=sha(marker)
            with verified(f,info['outputs'][f.name]).open() as z:
                rows.extend(csv.DictReader(z))
    assert len(rows)==2126
    csv_rows(OUT/'raw_by_sweep.csv',rows)
    keys=['raw',*LEDGER,'legacy_bright_region',*(f'fail_mask_{j}' for j in range(8))]
    parts={}
    for side in ('south','north'):
        sel=[x for x in rows if x['side']==side]
        d={k:sum(int(x[k]) for x in sel) for k in keys}
        assert d['raw']==sum(d[k] for k in LEDGER)
        assert d['outside_science_domain']==d['legacy_bright_region']+sum(d[f'fail_mask_{j}'] for j in range(8))
        parts[side]=d
    diagnostics=[]
    for side,d in parts.items():
        diagnostics.append(dict(side=side,rule='legacy_bright_region',first_failure=d['legacy_bright_region'],
                                anywhere=d['legacy_bright_region']))
        for bit,k in enumerate(('latitude','extinction','magnitude')):
            val=1<<bit
            work=[d[f'fail_mask_{j}'] for j in range(8) if j&val]
            defined=[d[f'fail_mask_{j}'] for j in range(8) if j&val and not j&(val-1)]
            diagnostics.append(dict(side=side,rule=k,first_failure=sum(defined),anywhere=sum(work)))
        assert sum(x['first_failure'] for x in diagnostics if x['side']==side)==d['outside_science_domain']
    csv_rows(OUT/'outside_cuts.csv',diagnostics)
    csv_rows(OUT/'raw_summary.csv',[dict(side=k,**v) for k,v in parts.items()])
    total={k:sum(x[k] for x in parts.values()) for k in keys}
    assert total['raw']==3967618454
    new=sum(total[k] for k in LEDGER[5:])
    assert new==64631542
    a=total['raw']-total['outside_tile']-total['other_release']
    stages=('Raw DR11 sweep rows','Native release and tile','Verified PSF','New domain parent',
            'Including frozen southern PSFs','Owned native parent','Complete distance PDFs','Core science')
    counts=(total['raw'],a,a-total['missing_type']-total['non_psf'],new,new+11246209,
            72289386,43347612,2281023)
    fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
    y=np.arange(len(stages))
    ax.barh(y,np.array(counts)-1e6,left=1e6,color=(BLUE,BLUE,BLUE,ORANGE,ORANGE,BLUE,BLUE,GREEN))
    ax.invert_yaxis()
    ax.set(xscale='log',yticks=y,yticklabels=stages,xlim=(1e6,2e10),
           xlabel='Recorded sources or input rows',title='From DR11 native sweeps to the Photdis science samples')
    for i,n in enumerate(counts):
        ax.text(n*1.06,i,f'{n:,}',va='center')
    save(fig,OUT,'F18_catalogue_flow',[dict(step=s,n=n) for s,n in zip(stages,counts)])


def main():
    require_compute()
    OUT.mkdir(exist_ok=True,parents=True)
    inputs={'style':check_templates()}
    a=reduce_sources(inputs)
    ar=random_areas(inputs)
    rows=[]
    for i,region in enumerate(REGIONS):
        for j,level in enumerate(LEVELS[1:]):
            for k,(lo,hi) in enumerate(zip(SHELLS[:-1],SHELLS[1:])):
                h,p=a['dist'][i,j,:,k]
                rows.append(dict(region=region,sample=level,lo_kpc=lo,hi_kpc=hi,median_bin=int(h),pdf_mass=float(p)))
    csv_rows(OUT/'distance_shells.csv',rows)
    rows=[]
    for i,region in enumerate(REGIONS):
        for j,level in enumerate(LEVELS):
            for k,n in enumerate(a['counts_by_pixel'][i,j]):
                if n or ar[i,0,k]:
                    rows.append(dict(region=region,sample=level,pixel=k,n=int(n),grz_deg2=ar[i,0,k],
                                     unmasked_deg2=ar[i,1,k]))
    csv_rows(OUT/'sky_pixels.csv',rows)
    atomic_npz(OUT/'atlas_bins.npz',**a,r_edges=np.array(R),color_edges=np.array(C),area_deg2=ar,
               nside=NSIDE)
    with plt.rc_context(STYLE):
        grid=sky_figures(a,ar)
        selected_rerender(grid,inputs)
        xyz_figure(a)
        strata_figure(a,ar)
        native_systems(a,ar)
        raw_flow(inputs)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,parts=28917,
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Render the selected census and area-controlled distance profiles in Pal 5 style."""
import json
import os

import healpy as hp
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import (BLUE, GRAY, GREEN, ORANGE, STYLE, aitoff, density,
    plt, render_reference_figures, save, spatial)
from v3_10_figures import check_templates, regression
from final_science_area import FIELDS
from final_science_catalog import C, DEST, REGIONS
from final_science_evidence import GAIA_STATES, NAMES, SHELLS, TIERS

LABELS = {'south_v35':'DECam high latitude','south_low':'DECam low-latitude windows',
          'north':'BASS/MzLS high latitude'}
DISPLAY = (('south_v35',1),('south_low',2),('north',2))
MAG = np.arange(14.,20.5,.5)
LAT = (0,15,30,40,50,60,75,90.01)
RESIDUAL_MAP = LinearSegmentedColormap.from_list("pal5_residual", (BLUE, "white", ORANGE))


def inputs():
    source = DEST/'evidence/complete.json'
    meta = json.loads(source.read_text())
    path = verified(source.with_name('selected.npz'),meta['outputs']['selected.npz'])
    with np.load(path,allow_pickle=False) as z:
        t = {k:z[k] for k in z.files}
    assert np.all(t['TYPE']=='PSF')
    areas = {}
    bindings = {'selected':sha(source),'style':check_templates()}
    for side in ('north','south'):
        p = DEST/'area'/side/'complete.json'
        q = json.loads(p.read_text())
        areas[side] = read_columns(verified(p.with_name('area.npz'),q['outputs']['area.npz']),
                                    ('effective_deg2',))['effective_deg2']
        bindings[f'area_{side}'] = sha(p)
    return t,areas,bindings


def science_region(t, region, tier):
    return (t['region']==REGIONS.index(region))&(t['science']==tier)


def gaia_magnitudes(dest,t):
    fig,axes = plt.subplots(1,3,figsize=(18,6),layout='constrained',sharey=False)
    rows = []
    gaia = t['gaia_state']
    groups = (('Reliable Gaia geometry',gaia==6,BLUE),
              ('Matched, below geometry requirement',np.isin(gaia,(3,4,5)),ORANGE),
              ('Missing or unresolved link',~np.isin(gaia,(3,4,5,6)),GRAY))
    for (region,tier),ax in zip(DISPLAY,axes):
        sel = science_region(t,region,tier)
        for status,k,color in groups:
            n = np.histogram(t['r'][sel&k],MAG)[0]
            ax.stairs(n,MAG,color=color,lw=1.5,label=status)
            rows.extend(dict(region=region,tier=TIERS[tier],gaia=status,r_min=a,r_max=b,n=int(v))
                        for a,b,v in zip(MAG[:-1],MAG[1:],n))
        band = 'Median below 2 kpc' if tier==1 else 'Median 2–5 kpc'
        ax.set(title=f'{LABELS[region]}\n{band}',xlabel='Observed r (mag)',ylabel='Sources per bin')
    axes[0].legend()
    fig.suptitle('Gaia geometric-distance coverage of the selected Photdis samples')
    save(fig,dest,'F09_gaia_increment',rows)


def references(dest,t):
    rs = t['r']-2.165*t['EBV']
    mags = np.column_stack((rs+t['color'],rs))
    main = (t['science']<=2)&t['heldout']
    cohorts = {}
    for name in NAMES:
        ref = t[name.lower()+'_kpc']
        w = main & np.isfinite(ref)&(ref>0)
        cohorts[name] = (ref[w],t['q'][w],mags[w])
    render_reference_figures(dest,cohorts,distance_title='Distance validation · selected near and 2–5 kpc samples')
    fig,axes = plt.subplots(2,3,figsize=(18,11),layout='constrained')
    rows = []
    bottom = ('r','rvs_feh','lat')
    z_edges = np.linspace(-2,2,81)
    data = {}
    for j,name in enumerate(NAMES):
        ref = t[name.lower()+'_kpc']
        use = (t['science']<=3)&t['heldout']&np.isfinite(ref)&(ref>0)
        dm = np.full(len(ref),np.nan)
        dm[use] = 5*np.log10(t['q'][use,2]/ref[use])
        abs_r = rs-10-5*np.log10(np.where(use,ref,np.nan))
        ax = axes[0,j]
        xedge,yedge = np.linspace(-.1,1.8,64),np.linspace(-2.,12.,61)
        counts = np.histogram2d(t['color'][use],abs_r[use],(xedge,yedge))[0]
        totals = np.histogram2d(t['color'][use],abs_r[use],(xedge,yedge),weights=dm[use])[0]
        means = np.divide(totals,counts,out=np.full_like(totals,np.nan),where=counts>=5)
        m = ax.pcolormesh(xedge,yedge,means.T,cmap=RESIDUAL_MAP,vmin=-1,vmax=1,rasterized=True)
        fig.colorbar(m,ax=ax,pad=.02,label='Mean modulus residual (mag)')
        ax.set(title=f'{name} · reference CMD',xlabel='Dereddened g − r (mag)',
               ylabel='Reference absolute r (mag)',ylim=(12,-2))
        k = bottom[j]
        x = t[k] if k!='lat' else abs(t[k])
        edges = (MAG,np.linspace(-3.,.5,57),np.linspace(0.,90.,46))[j]
        w = use.copy()
        if k=='rvs_feh':
            w &= (t['rvs_feh_err']>0)&(t['rvs_feh_err']<=.3)
        ax = axes[1,j]
        hist = density(fig,ax,x[w],dm[w],edges,z_edges)
        ax.axhline(0,ls='--',color=ORANGE,lw=1.25)
        ax.set(title=name,xlabel=('Observed r (mag)','Spectroscopic [Fe/H] (dex)',
            'Absolute Galactic latitude (deg)')[j],ylabel='Distance modulus residual (mag)')
        data.update({f'color_{j}':xedge,f'abs_{j}':yedge,f'cmd_n_{j}':counts,
                     f'cmd_dmu_sum_{j}':totals,f'axis_{j}':edges,f'residual_{j}':hist})
        rows.append(dict(reference=name,cmd_stars=int(use.sum()),lower_panel=k,
                         lower_stars=int(w.sum()),lower_plotted=int(hist.sum())))
    fig.suptitle('Photometric distance residuals by luminosity branch and observing domain')
    save(fig,dest,'F10_distance_residuals',rows,dict(residual_edges=z_edges,**data))


def footprint(t):
    l = -np.deg2rad((t['lon']+180)%360-180)
    b = np.deg2rad(t['lat'])
    le,be = np.linspace(-np.pi,np.pi,145),np.linspace(-np.pi/2,np.pi/2,73)
    h = np.histogram2d(l,b,bins=(le,be))[0].T
    return l,b,(.5*(le[1:]+le[:-1]),.5*(be[1:]+be[:-1]),h>0)


def regional_sky(dest,t,areas):
    d = t['q'][:,2]
    l,b,fp = footprint(t)
    primary = t['science']<=2
    atlas = t['science']<=3
    aitoff(dest,'F04_science','Selected Photdis sources · 5–10 kpc is a research tier',
           d,l,b,atlas,fp,BLUE)
    aitoff(dest,'F04_weak_gaia','Gaia counterpart below geometric-distance cut · 5–10 kpc is a research tier',
           d,l,b,atlas&np.isin(t['gaia_state'],(3,4,5)),fp,BLUE)
    ref = t['gaia_kpc'].copy()
    for k in ('specdis_kpc','spdist_kpc'):
        ref = np.where(np.isfinite(ref),ref,t[k])
    spatial(dest,{'primary':primary},d,l,b,ref,np.isfinite(ref)&t['heldout'])
    nside = C['sky_nside']
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside,t['RA'],t['DEC'],lonlat=True,nest=True)
    ra,dec = hp.pix2ang(nside,np.arange(npix),lonlat=True,nest=True)
    _,b_center = galactic(ra,dec)
    hardshell = np.searchsorted(SHELLS,d,side='right')-1
    rows,summary = [],[]
    f,axes = plt.subplots(1,3,figsize=(18,6),layout='constrained')
    for (region,tier),ax in zip(DISPLAY,axes):
        site = 'north' if region=='north' else 'south'
        zone = int(region!='south_low')
        ar = areas[site][zone]
        # A common three-band depth threshold for each comparison.
        depth = 19.4 if tier==1 else 21.
        area = ar[FIELDS.index(f'grz_{depth:g}')]
        use = science_region(t,region,tier)
        total = np.bincount(pix[use],minlength=npix)
        control = (ar[1]>0)&(area>=.9*ar[1])&(area>=.5)
        control &= (abs(b_center)>30)==bool(zone)
        for k in range(3):
            hard = np.bincount(pix[use&(hardshell==k)],minlength=npix)
            soft = np.bincount(pix[use],weights=t['bins'][use,k],minlength=npix)
            good = use&np.isin(t['gaia_state'],(3,4,5))
            increment = np.bincount(pix[good],weights=t['bins'][good,k],minlength=npix)
            records = np.flatnonzero((area>0)|(total>0))
            rows.extend(dict(region=region,tier=TIERS[tier],pixel=int(i),area_deg2=float(area[i]),
                             depth_r_g_z=depth,pixel_pass=bool(control[i]),distance_lo=SHELLS[k],
                             distance_hi=SHELLS[k+1],selected=int(total[i]),hard=int(hard[i]),
                             pdf=float(soft[i]),weak_gaia_pdf=float(increment[i])) for i in records)
            for a,c in zip(LAT[:-1],LAT[1:]):
                j = control&(abs(b_center)>=a)&(abs(b_center)<c)
                if not j.any():
                    continue
                aa = area[j].sum()
                summary.append(dict(region=region,tier=TIERS[tier],abs_b_lo=a,abs_b_hi=c,
                    shell=k,pixels=int(j.sum()),area_deg2=float(aa),selected=int(total[j].sum()),
                    hard=int(hard[j].sum()),pdf=float(soft[j].sum()),
                    pdf_density=float(soft[j].sum()/aa),hard_density=float(hard[j].sum()/aa)))
        k = 0 if tier==1 else 1
        view = [r for r in summary if r['region']==region and r['shell']==k]
        x = np.array([.5*(r['abs_b_lo']+r['abs_b_hi']) for r in view])
        ax.plot(x,[r['hard_density'] for r in view],'o-',color=BLUE,label='Median-distance bins')
        ax.plot(x,[r['pdf_density'] for r in view],'s--',color=ORANGE,label='Full PDF mass')
        ax.set(title=f'{LABELS[region]}\n{SHELLS[k]:g}–{SHELLS[k+1]:g} kpc',
               xlabel='Absolute Galactic latitude (deg)',ylabel='Selected sources per deg²')
    axes[0].legend()
    f.suptitle('Latitude profiles on a common-depth footprint within each survey domain')
    save(f,dest,'F11_depth_controlled_profiles',summary)
    csv_rows(dest/'pixel_selection.csv',rows)


def main():
    require_compute()
    dest = DEST/'figures'
    dest.mkdir(exist_ok=True,parents=True)
    t,areas,binding = inputs()
    with plt.rc_context(STYLE):
        baseline = dest/'baseline'
        baseline.mkdir(exist_ok=True)
        check = regression(baseline)
        gaia_magnitudes(dest,t)
        references(dest,t)
        regional_sky(dest,t,areas)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),rows=len(t['ls_id']),
        inputs=binding,regression=check,outputs={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='complete.json'}))


if __name__=='__main__':
    main()

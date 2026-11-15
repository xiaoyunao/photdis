#!/usr/bin/env python3
"""Verify the full partition set and measure two selections on the same footprint."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np

from photdis.stage1 import galactic, require_compute, tile_bounds
from v2_1_near import atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_area import AREA, DEPTH, NSIDE
from v3_5_dr11_inventory import OUT
from v3_4_near_forecast import qrf_bins
from v3_5_dr11_pilot import LEDGER, in_tile
from v3_5_dr11_production import FREEZE, INDEX, PRODUCTION, check_pdf

DELIVERY = OUT/'delivery'
SHELLS = ('0–2 kpc','2–5 kpc','5–10 kpc')
CUTS = ('Primary selection','Strict selection')
REGIONS = ('all','v34_photometric_analogs','faint_blue')


def audit_parts(dest, name, parts, status, ids, counts):
    """Read the delivered columns and reconcile them with the parent ledger."""
    bounds = tile_bounds(name)
    offset, parent_rows = 0, []
    observed = Counter()
    for file in parts:
        with np.load(dest/file,allow_pickle=False) as z:
            q = {k:z[k] for k in ('TYPE','qrf_supported','primary','strict','qrf_mu_mass',
                                   'qrf_mu_edges','qrf_bins','qrf_qnear','qrf_quantiles_kpc')}
            check_pdf(q)
            distance = q['qrf_quantiles_kpc']
            if (not np.isfinite(distance).all() or (distance<=0).any()
                    or (np.diff(distance,axis=1)<0).any()):
                raise ValueError(f'Invalid distance quantiles: {name}/{file}')
            np.testing.assert_allclose(qrf_bins(q['qrf_mu_edges'],q['qrf_mu_mass']),q['qrf_bins'],
                                       atol=1e-12,rtol=0)
            if not in_tile(z['RA'],z['DEC'],bounds).all() or not (z['photometric_system']=='DECam_DR11').all():
                raise ValueError(f'Wrong sweep ownership or photometric system: {name}/{file}')
            n = len(q['TYPE'])
            for key in ('ls_id','source_id'):
                np.testing.assert_array_equal(z[key],ids[key][offset:offset+n])
            parent_rows.append(z['parent_row'])
            offset += n
            observed.update(rows=n,primary=int(q['primary'].sum()),strict=int(q['strict'].sum()),
                            reused=int(z['reused_v34'].sum()),gate=int(z['nominal_evaluated'].sum()),
                            numerical=int(z['nominal_numerical_flag'].sum()))
    parent = np.concatenate(parent_rows) if parent_rows else np.empty(0,dtype='i8')
    np.testing.assert_array_equal(np.sort(parent),np.flatnonzero(status==len(LEDGER)-1))
    if observed != Counter(counts):
        raise ValueError(f'Exported partition totals disagree with receipt: {name}')


def read_tile(name, plan_digest):
    dest = PRODUCTION/Path(name).stem
    marker = dest/'complete.json'
    m = json.loads(marker.read_text())
    if m['binding']['freeze_sha256']!=plan_digest or m['source']!=name or not m['all_PSF']:
        raise ValueError(f'Wrong sweep provenance: {name}')
    parts = [f for f in m['outputs'] if f.startswith('part_')]
    if set(parts)!=set(p.name for p in dest.glob('part_*.npz')):
        raise ValueError(f'Unexpected sweep file set: {name}')
    for f,digest in m['outputs'].items():
        verified(dest/f,digest)
    status = load(dest/'parent_status.npz')['status']
    np.testing.assert_array_equal(np.bincount(status,minlength=len(m['exclusive_counts'])),
                                  list(m['exclusive_counts'].values()))
    ids = load(dest/'identities.npz')
    space = load(dest/'spatial.npz')
    np.testing.assert_allclose(space['mass'].sum(axis=2),space['selected'],atol=1e-7,rtol=0)
    if len(ids['ls_id'])!=m['counts'].get('rows',0) or len(np.unique(ids['ls_id']))!=len(ids['ls_id']):
        raise ValueError(f'Incomplete unique identity list: {name}')
    audit_parts(dest,name,sorted(parts),status,ids,m['counts'])
    return m,ids,space,sha(marker),len(parts)


def angular_bins(mass, area, deep, used):
    n = len(area)
    ra,dec = hp.pix2ang(NSIDE,np.arange(n),nest=True,lonlat=True)
    lon,b = galactic(ra,dec)
    result = []
    for wide,name in enumerate(CUTS):
        for region,rname in enumerate(REGIONS):
            for lo,hi in ((30,45),(45,60),(60,90)):
                for hemi, sign in (('north',1),('south',-1)):
                    for i in range(4):
                        domain = (b*sign>=lo)&(b*sign<hi)&(lon>=i*90)&(lon<(i+1)*90)&used
                        wide_area = area[domain].sum()
                        if wide_area<=0:
                            continue
                        dm = mass[wide,region,:3,domain].sum(axis=0)
                        for shell,value in enumerate(dm):
                            result.append(dict(selection=name,cohort=rname,latitude_min=lo,latitude_max=hi,
                                hemisphere=hemi,longitude_min=i*90,longitude_max=(i+1)*90,shell=SHELLS[shell],
                                area_deg2=wide_area,area_r20_deg2=deep[domain].sum(),
                                posterior_mass=value,per_deg2=value/wide_area))
    return result



def shell_comparison(mass,area,deep,limiting_mag=None):
    """Compare spatial shape on a common imaging area under both selections."""
    radius = hp.nside2pixarea(NSIDE,degrees=True)
    masks = {'imaged':area>0, 'quarter_pixel':area>=radius/4}
    if limiting_mag is not None:
        px = np.flatnonzero(masks['quarter_pixel']&np.isfinite(limiting_mag))
        ordered = px[np.argsort(limiting_mag[px],kind='stable')]
        cumulative = np.cumsum(area[ordered])
        split = np.searchsorted(cumulative,cumulative[-1]/2)+1 if len(cumulative) else 0
        if split==0 or split==len(ordered):
            raise ValueError('The covered sky has no usable depth split')
        masks['shallower_r'] = np.isin(np.arange(len(area)),ordered[:split])
        masks['deeper_r'] = np.isin(np.arange(len(area)),ordered[split:])
    ra,dec = hp.pix2ang(NSIDE,np.arange(len(area)),nest=True,lonlat=True)
    lon,b = galactic(ra,dec)
    result = []
    for footprint,mask in masks.items():
        for case,region in enumerate(REGIONS):
            for shell,label in enumerate(SHELLS):
                q = mass[:,case,shell][:,mask]
                sums = q.sum(axis=1)
                if not (sums>0).all():
                    continue
                total_area = area[mask].sum()
                tv = .5*np.abs(q[0]/sums[0]-q[1]/sums[1]).sum()
                # Match the four longitude sectors across hemispheres before
                # forming a contrast. Each sector contributes its shared area.
                hemisphere,matched_area = np.zeros((2,2)),0.
                for start in (0.,90.,180.,270.):
                    sector = mask&(lon>=start)&(lon<start+90)
                    halves = [sector&(b>=0), sector&(b<0)]
                    sizes = np.asarray([area[h].sum() for h in halves])
                    share = sizes.min()
                    if share>0:
                        hemisphere += np.column_stack(
                            [mass[:,case,shell][:,h].sum(axis=1)/size for h,size in zip(halves,sizes)])*share
                        matched_area += share
                nr,sr = np.divide(hemisphere[:,0],hemisphere[:,1],out=np.full(2,np.nan),
                                  where=hemisphere[:,1]>0)
                depth = {}
                if limiting_mag is not None:
                    measured = mask&np.isfinite(limiting_mag)
                    depth['mean_r_limit_mag'] = np.average(limiting_mag[measured],weights=area[measured])
                result.append(dict(footprint=footprint,cohort=region,shell=label,area_deg2=total_area,**depth,
                    depth_fraction=deep[mask].sum()/total_area,primary_mass=sums[0],strict_mass=sums[1],
                    primary_per_deg2=sums[0]/total_area,strict_per_deg2=sums[1]/total_area,
                    retained_fraction=sums[1]/sums[0],spatial_total_variation=tv,
                    longitude_matched_area_per_hemisphere_deg2=matched_area,
                    primary_north_south_ratio=nr,strict_north_south_ratio=sr))
    return result


def read_depth(coverage,area_marker,area):
    marker = DEPTH/'complete.json'
    receipt = json.loads(marker.read_text())
    if receipt['frozen_area_sha256']!=sha(area_marker):
        raise ValueError('Depth is not bound to the delivered imaging area')
    name = 'sweep_healpix_depth.npz'
    q = load(verified(DEPTH/name,receipt['outputs'][name]))
    for key in ('sweep','pixel'):
        np.testing.assert_array_equal(q[key],area[key])
    usable,weighted = [np.bincount(q['pixel'][coverage],weights=q[k][coverage],minlength=hp.nside2npix(NSIDE))
                       for k in ('valid_area_deg2','limiting_mag_times_area')]
    return np.divide(weighted,usable,out=np.full(len(usable),np.nan),where=usable>0), sha(marker)

def maps(mass,area):
    ra,dec = hp.pix2ang(NSIDE,np.arange(len(area)),nest=True,lonlat=True)
    l,b = galactic(ra,dec)
    x,y = np.deg2rad((-l+180)%360-180),np.deg2rad(b)
    valid = area>0
    fig,axes = plt.subplots(2,3,figsize=(13,6.4),subplot_kw={'projection':'mollweide'},
                             layout='constrained')
    for shell in range(3):
        density = mass[:,0,shell,valid]/area[valid]
        positive = density[density>0]
        if len(positive):
            low, high = np.percentile(positive,[5,98])
            norm = LogNorm(max(low,1e-6),max(high,1.01*max(low,1e-6)))
        else:
            norm = None
        for j in range(2):
            ax = axes[j,shell]
            ax.scatter(x[valid],y[valid],s=8,c='0.84',rasterized=True)
            im = ax.scatter(x[valid],y[valid],c=np.ma.masked_less_equal(density[j],0),s=8,
                            cmap='viridis',norm=norm,rasterized=True)
            ax.set_longitude_grid(60)
            ax.set_latitude_grid(30)
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda x,_:f'{(-round(np.rad2deg(x)))%360}°'))
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.18)
            ax.set_title(f'{CUTS[j]} · {SHELLS[shell]}',fontsize=10)
            if j == 1:
                ax.set_xlabel('Galactic longitude')
            if shell == 0:
                ax.set_ylabel('Galactic latitude')
        ticks = np.geomspace(norm.vmin,norm.vmax,4) if norm else None
        bar = fig.colorbar(im,ax=axes[:,shell],orientation='horizontal',shrink=.75,
                           extend='both' if norm else 'neither',ticks=ticks,
                           format=FuncFormatter(lambda value,_:f'{value:.2g}'),
                           label='Posterior-weighted sources per deg²')
        bar.ax.xaxis.set_minor_locator(NullLocator())
    for ext in ('png','pdf'):
        fig.savefig(DELIVERY/f'spatial_shells.{ext}',dpi=260)
    plt.close(fig)


def main():
    require_compute()
    plan = json.loads(FREEZE.read_text())
    digest = sha(FREEZE)
    names = [name for group in plan['groups'] for name in group]
    if len(set(names))!=len(names):
        raise ValueError('Frozen sweep roster contains duplicates')
    inv_path = OUT/'inventory/complete.json'
    sweeps = json.loads(inv_path.read_text())['sweeps']
    inventory = {s['filename']:i for i,s in enumerate(sweeps)}
    area_marker = AREA/'complete.json'
    area_receipt = json.loads(area_marker.read_text())
    area = load(verified(AREA/'sweep_healpix.npz',area_receipt['outputs']['sweep_healpix.npz']))
    npix = hp.nside2npix(NSIDE)
    coverage = np.isin(area['sweep'],[inventory[n] for n in names])
    a,deep = [np.bincount(area['pixel'][coverage],weights=area[k][coverage],minlength=npix)
              for k in ('area_deg2','area_r20_deg2')]
    limiting_mag,depth_digest = read_depth(coverage,area_marker,area)
    ids,native,receipts = [],[],[]
    totals, attrition = Counter(),Counter()
    draws = np.zeros((2,3,6,npix))
    task_cpus = int(os.getenv('SLURM_CPUS_PER_TASK','8'))
    with ThreadPoolExecutor(max_workers=min(task_cpus,16)) as pool:
        for m,x,q,h,n in pool.map(lambda name:read_tile(name,digest),names):
            ids.append(x['ls_id'])
            native.append(x['source_id'])
            totals.update(m['counts'])
            attrition.update(m['exclusive_counts'])
            draws += q['mass']
            receipts.append(dict(sweep=m['source'],receipt_sha256=h,partitions=n,**m['counts']))
    merged = np.concatenate(ids)
    if len(np.unique(merged))!=len(merged) or len(merged)!=totals['rows']:
        raise ValueError('Global Legacy identities are not unique or complete')
    before = load(verified(INDEX,plan['archive_index_sha256']))['ls_id']
    in_current = int(np.isin(before,merged).sum())
    if in_current != totals['reused']:
        raise ValueError('The frozen intersection of native v3.4 rows was not reused exactly')
    aliases = np.concatenate(native)
    aliases = aliases[aliases>0]
    distinct,nmatch = np.unique(aliases,return_counts=True)
    np.testing.assert_allclose(draws[:,0].sum(axis=(1,2)),[totals['primary'],totals['strict']],atol=1e-6,rtol=0)
    DELIVERY.mkdir(parents=True,exist_ok=True)
    csv_rows(DELIVERY/'sweeps.csv',receipts)
    csv_rows(DELIVERY/'angular_density.csv',angular_bins(draws,a,deep,a>0))
    csv_rows(DELIVERY/'shell_comparison.csv',shell_comparison(draws,a,deep,limiting_mag))
    atomic_npz(DELIVERY/'spatial.npz',mass=draws,area_deg2=a,area_r20_deg2=deep,
               r_limiting_mag=limiting_mag,ambiguous_gaia_aliases=distinct[nmatch>1])
    csv_rows(DELIVERY/'shell_mass.csv',[dict(selection=cut,cohort=region,shell=shell,
             model_count=draws[i,j,k].sum(),uncovered_mass=draws[i,j,k,a<=0].sum())
             for i,cut in enumerate(CUTS) for j,region in enumerate(REGIONS) for k,shell in enumerate(SHELLS)])
    maps(draws,a)
    save_json(DELIVERY/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),production_freeze_sha256=digest,
        depth_profile_sha256=depth_digest,
        sweeps=len(receipts),parts=sum(r['partitions'] for r in receipts),all_PSF=True,
        totals=totals,exclusions=attrition,absent_bad_headers=plan['excluded_bad_headers'],
        archived_v34_rows=len(before),archived_v34_intersection=in_current,
        archived_v34_outside_current=len(before)-in_current,
        ambiguous_gaia_alias_n=int((nmatch>1).sum()),area_deg2=a.sum(),area_r20_deg2=deep.sum(),
        outputs={p.name:sha(p) for p in sorted(DELIVERY.iterdir()) if p.name!='complete.json'}))


if __name__=='__main__':
    main()

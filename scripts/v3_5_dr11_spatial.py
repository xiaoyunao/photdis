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
import numpy as np

from photdis.stage1 import galactic, require_compute
from v2_1_near import atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_area import AREA, NSIDE
from v3_5_dr11_inventory import OUT
from v3_5_dr11_production import FREEZE, INDEX, PRODUCTION

DELIVERY = OUT/'delivery'
SHELLS = ('0–2 kpc','2–5 kpc','5–10 kpc')
CUTS = ('Primary selection','Strict selection')
REGIONS = ('all','v34_photometric_analogs','faint_blue')


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


def maps(mass,area):
    ra,dec = hp.pix2ang(NSIDE,np.arange(len(area)),nest=True,lonlat=True)
    l,b = galactic(ra,dec)
    x,y = np.deg2rad((180-l+180)%360-180),np.deg2rad(b)
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
            im = ax.scatter(x[valid],y[valid],c=np.ma.masked_less_equal(density[j],0),s=8,
                            cmap='viridis',norm=norm,rasterized=True)
            ax.grid(alpha=.18)
            ax.set_title(f'{CUTS[j]} · {SHELLS[shell]}',fontsize=10)
            ax.set_xlabel('Galactic longitude')
            ax.set_ylabel('Galactic latitude')
        fig.colorbar(im,ax=axes[:,shell],orientation='horizontal',shrink=.75,
                     label='Posterior-weighted sources per square degree')
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
    area_marker = json.loads((AREA/'complete.json').read_text())
    area = load(verified(AREA/'sweep_healpix.npz',area_marker['outputs']['sweep_healpix.npz']))
    npix = hp.nside2npix(NSIDE)
    coverage = np.isin(area['sweep'],[inventory[n] for n in names])
    a,deep = [np.bincount(area['pixel'][coverage],weights=area[k][coverage],minlength=npix)
              for k in ('area_deg2','area_r20_deg2')]
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
    atomic_npz(DELIVERY/'spatial.npz',mass=draws,area_deg2=a,area_r20_deg2=deep,
               ambiguous_gaia_aliases=distinct[nmatch>1])
    csv_rows(DELIVERY/'shell_mass.csv',[dict(selection=cut,cohort=region,shell=shell,
             model_count=draws[i,j,k].sum(),uncovered_mass=draws[i,j,k,a<=0].sum())
             for i,cut in enumerate(CUTS) for j,region in enumerate(REGIONS) for k,shell in enumerate(SHELLS)])
    maps(draws,a)
    save_json(DELIVERY/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),production_freeze_sha256=digest,
        sweeps=len(receipts),parts=sum(r['partitions'] for r in receipts),all_PSF=True,
        totals=totals,exclusions=attrition,absent_bad_headers=plan['excluded_bad_headers'],
        archived_v34_rows=len(before),archived_v34_intersection=in_current,
        archived_v34_outside_current=len(before)-in_current,
        ambiguous_gaia_alias_n=int((nmatch>1).sum()),area_deg2=a.sum(),area_r20_deg2=deep.sum(),
        outputs={p.name:sha(p) for p in sorted(DELIVERY.iterdir()) if p.name!='complete.json'}))


if __name__=='__main__':
    main()

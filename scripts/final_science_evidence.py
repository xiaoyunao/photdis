#!/usr/bin/env python3
"""Audit Gaia availability and compare observed reference distances on owned PSFs."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os

import numpy as np

from photdis.gaia import NPIX, SOURCE_ID_HEALPIX_FACTOR, derive_gaia_labels, join_gaia
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_4_near_validate import GAIA, tile_inventory
from v3_5_dr11_production import CUTS
from v3_6_gaia_census import STATES, distances
from v3_6_vac_distances import read_columns
from v3_8_candidate import read_cache
from v3_9_figures import evaluation_rows
from v3_9_references import join, matched, raw_catalogs, subset
from v3_10_evidence import rvs_evidence
from final_science_catalog import C, CONFIG, CENSUS_FIELDS, DEST, REGIONS, indices

GAIA_STATES = (*STATES,'archive_tile_missing','ambiguous_native_gaia_identity')
SHELLS = (0.,2.,5.,10.,20.,50.,np.inf)
TIERS = ('','bright_red_near','dwarf_enhanced_2_5','dwarf_enhanced_5_10_research',
         'dwarf_enhanced_other_median')
NAMES = ('Gaia','SpecDis','SPDist')
SUMMARY_KEYS = ('n','median_relative','nmad_relative','median_mu_residual','severe_fraction',
                'coverage68','coverage95','median_width68','median_width95')


def read_scan(inputs, dest):
    _, _, expected = indices()
    total = {}
    parts = []
    scanned = 0
    for i in range(C['scan_shards']):
        marker = DEST/'scan'/f'{i:02d}.json'
        receipt = json.loads(marker.read_text())
        for key,value in expected.items():
            if receipt['inputs'][key]!=value:
                raise ValueError(f'Changed binding {key} in {marker}')
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
        scanned += receipt['parts']
        for name,digest in receipt['outputs'].items():
            p = verified(marker.parent/name,digest)
            if name.endswith('.npz'):
                parts.append(read_columns(p,('TYPE','ls_id','source_id','RA','DEC','EBV','r','color',
                                              'lon','lat','q','bins','science','region')))
            else:
                with p.open() as f:
                    for r in csv.DictReader(f):
                        k = tuple(r[c] for c in CENSUS_FIELDS[:-1])
                        total[k] = total.get(k,0)+int(r['n'])
    csv_rows(dest/'catalog_census.csv',[dict(zip(CENSUS_FIELDS,(*k,v))) for k,v in sorted(total.items())])
    aggregates = []
    receipt = json.loads((ROOT/'data/v3_9/delivery/complete.json').read_text())['unified']
    for region in REGIONS:
        n = sum(v for k,v in total.items() if k[0]==region)
        pdf = sum(v for k,v in total.items() if k[0]==region and k[-1]=='3')
        strict_pdf = sum(v for k,v in total.items() if k[0]==region and k[-2:] == ('7','3'))
        if (n,pdf)!=(receipt[region]['owned_rows'],receipt[region]['owned_pdf_rows']):
            raise ValueError(f'Census does not close in {region}')
        aggregates.append(dict(region=region,owned=n,pdf=pdf,strict_pdf=strict_pdf,research_pdf=pdf-strict_pdf))
    if scanned!=sum(r['parts'] for r in receipt.values()):
        raise ValueError('Incomplete partition scan')
    csv_rows(dest/'catalog_totals.csv',aggregates)
    t = join(parts)
    if len(np.unique(t['ls_id']))!=len(t['ls_id']) or not np.all(t['TYPE']=='PSF'):
        raise ValueError('Selected rows must have unique native PSF identities')
    work = json.loads((DEST/'working_ids.json').read_text())
    ids = read_columns(verified(DEST/'working_ids.npz',work['outputs']['working_ids.npz']),('south','north'))
    np.testing.assert_array_equal(np.sort(t['ls_id'][t['science']>=2]),np.sort(np.r_[ids['south'],ids['north']]))
    return t


def classify_gaia(t, dest):
    ids = t['source_id']
    valid = (ids>0)&(ids<NPIX*SOURCE_ID_HEALPIX_FACTOR)
    cfg = json.loads(CUTS.read_text())
    inventory_path = dest/'gaia_tiles.json'
    inventory = tile_inventory(ids[valid])
    save_json(inventory_path,inventory)
    k = np.where(valid,ids//SOURCE_ID_HEALPIX_FACTOR//384,32)
    order = np.argsort(k,kind='stable')
    groups = np.split(order,np.flatnonzero(np.diff(k[order]))+1)

    def run(j):
        p = subset(t,j)
        gaia = join_gaia(np.where(valid[j],ids[j],0),GAIA)
        a = derive_gaia_labels(gaia)
        state,d,lo,hi = distances(p,valid[j],a,gaia,cfg)
        state[gaia['gaia_file_missing']] = 7
        return j,state,d,lo,hi,a['p_corr']

    status = np.zeros(len(ids),'u1')
    q = np.full((4,len(ids)),np.nan)
    with ThreadPoolExecutor(max_workers=int(os.getenv('SLURM_CPUS_PER_TASK','1'))) as pool:
        for j,state,*out in pool.map(run,groups):
            status[j] = state
            q[:,j] = out
    associated = (status>=3)&(status<=6)
    keys,counts = np.unique(ids[associated],return_counts=True)
    ambiguous = associated&np.isin(ids,keys[counts>1])
    status[ambiguous] = 8
    q[:3,ambiguous] = np.nan
    if inventory!=tile_inventory(ids[valid]):
        raise ValueError('Gaia archive changed during reduction')
    t.update(gaia_state=status,gaia_kpc=q[0],gaia_lo=q[1],gaia_hi=q[2],gaia_signed=q[3])


def add_references(t,inputs):
    catalogs,meta = raw_catalogs()
    inputs.update({f'raw_{k}':v for k,v in meta.items()})
    for name in ('specdis','spdist'):
        raw = catalogs[name]
        row = matched(t,raw)
        valid = row>=0
        valid[valid] &= raw['historical_quality'][row[valid]]&raw['usable_uncertainty'][row[valid]]
        keys,counts = np.unique(t['source_id'][valid],return_counts=True)
        valid &= ~np.isin(t['source_id'],keys[counts>1])
        d = np.full(len(row),np.nan)
        d[valid] = raw['distance_kpc'][row[valid]]
        t[name+'_kpc'] = d
    archive = rvs_evidence(t,inputs)
    t.update({k:archive[k] for k in ('rvs_label','rvs_feh','rvs_feh_err','rvs_logg','rvs_logg_err')})
    fit = read_cache(ROOT/'data/v3_0/controls','training.npz')
    inputs[str(fit.relative_to(ROOT))] = sha(fit)
    training = read_columns(fit,('identity',))['identity']
    nr = t['region']==REGIONS.index('north')
    t['heldout'] = np.where(nr,evaluation_rows('north',t,None),evaluation_rows('south',t,training))


def profiles(t, dest):
    rows = []
    typ = t['science']
    h = np.searchsorted(SHELLS,t['q'][:,2],side='right')-1
    mag = np.searchsorted(C['magnitude_edges'],t['r'],side='right')-1
    for ri,region in enumerate(REGIONS):
        for tier in np.unique(typ[t['region']==ri]):
            domain = (t['region']==ri)&(typ==tier)
            for m in (-1,*np.unique(mag[domain])):
                eligible = domain & ((mag==m) if m>=0 else True)
                for k,name in enumerate(GAIA_STATES):
                    keep = eligible&(t['gaia_state']==k)
                    hard = np.bincount(h[keep],minlength=len(SHELLS)-1)
                    soft = t['bins'][keep].sum(axis=0,dtype='f8')
                    np.testing.assert_allclose(soft.sum(),keep.sum(),rtol=0,atol=2e-6)
                    rows.append(dict(region=region,tier=TIERS[tier],magnitude_bin=m,gaia_state=name,n=int(keep.sum()),
                                     **{f'hard_{j}':v for j,v in enumerate(hard)},
                                     **{f'pdf_{j}':v for j,v in enumerate(soft)}))
    csv_rows(dest/'gaia_increment.csv',rows)
    reference = []
    lat = abs(t['lat'])
    c = t['color']
    feh = t['rvs_feh']
    q = 10+5*np.log10(t['q'])
    rprof = [('all','all',np.ones(len(typ),bool))]
    for axis,val,edges in (('r',t['r'],C['magnitude_edges']),
                           ('color',c,C['color_edges']),
                           ('abs_latitude',lat,C['abs_latitude_edges']),
                           ('feh',feh,(-10,-2.,-1.,-.5,2.))):
        for a,b in zip(edges[:-1],edges[1:]):
            w = (val>=a)&(val<b)
            if axis=='feh':
                w &= (t['rvs_feh_err']>0)&(t['rvs_feh_err']<=.3)
            rprof.append((axis,f'{a}:{b}',w))
    for lab in (-1,0,1,2):
        rprof.append(('rvs_class',str(lab),t['rvs_label']==lab))
    for source in NAMES:
        d = t[source.lower()+'_kpc']
        mu = np.full(len(d),np.nan)
        good = np.isfinite(d)&(d>0)
        mu[good] = 10+5*np.log10(d[good])
        for ri,region in enumerate(REGIONS):
            for tier in np.unique(typ[t['region']==ri]):
                base = (t['region']==ri)&(typ==tier)&good
                for lo,hi in zip(SHELLS[:-1],SHELLS[1:]):
                    fit = base & (d>=lo)&(d<hi)
                    for split,bin_name,mask in rprof:
                        elig = fit&mask
                        j = elig&t['heldout']
                        metric = distance_summary(q[j],mu[j])
                        reference.append(dict(region=region,tier=TIERS[tier],source=source,ref_min=lo,
                            ref_max=hi,axis=split,bin=bin_name,matched=int(elig.sum()),
                            **{key:metric.get(key) for key in SUMMARY_KEYS}))
    csv_rows(dest/'heldout_distance.csv',reference)


def main():
    require_compute()
    dest = DEST/'evidence'
    dest.mkdir(exist_ok=True,parents=True)
    inputs = {'config':sha(CONFIG),'script':sha(ROOT/'scripts/final_science_evidence.py'),'gaia_cut':sha(CUTS)}
    t = read_scan(inputs,dest)
    classify_gaia(t,dest)
    add_references(t,inputs)
    profiles(t,dest)
    atomic_npz(dest/'selected.npz',**t)
    names = ('catalog_census.csv','catalog_totals.csv','gaia_increment.csv','heldout_distance.csv',
             'selected.npz','gaia_tiles.json')
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,rows=len(t['ls_id']),
                      outputs={name:sha(dest/name) for name in names}))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Census of owned rows and exact replay of evidence-backed catalog selections."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os

import healpy as hp
import numpy as np

from photdis.stage1 import galactic, require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_delivery import DEST as RELEASE
from v3_9_references import join, subset
from v3_9_unified import owner_rows
from v3_10_figures import check_templates
from v3_10_science import working_index

DEST = ROOT/'data/final_science'
CONFIG = ROOT/'configs/final_science.json'
C = json.loads(CONFIG.read_text())
REGIONS = ('south_v35', 'south_faint', 'south_low', 'north')
DEPTH_FIELDS = ('PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z')
CENSUS_FIELDS = ('region','r_bin','color_bin','lat_bin','ebv_bin','quality','state','n')


def indices():
    release = RELEASE/'complete.json'
    meta = json.loads(release.read_text())
    index = verified(RELEASE/'unified_parts.csv',meta['outputs']['unified_parts.csv'])
    seam = RELEASE/'seam/complete.json'
    sm = json.loads(verified(seam,meta['sources'][str(seam.relative_to(ROOT))]).read_text())
    name = 'ownership_overrides.npz'
    overrides = read_columns(verified(seam.parent/name,sm['outputs'][name]),
                            ('ls_id','TYPE','primary_survey_owner'))
    with index.open() as f:
        parts = list(csv.DictReader(f))
    return parts,overrides,{'release':sha(release),'index':sha(index),'seam':sha(seam),
                           'config':sha(CONFIG),'style':check_templates()}


def histogram(region, r, color, lat, ebv, quality, state):
    args = (r,color,abs(lat),ebv)
    names = ('magnitude','color','abs_latitude','ebv')
    bins = [np.searchsorted(C[f'{k}_edges'],v,side='right')-1 for k,v in zip(names,args)]
    for i,(v,name) in enumerate(zip(args,names)):
        named = C[f'{name}_edges']
        bins[i] = np.where(np.isfinite(v)&(v>=named[0])&(v<named[-1]),bins[i],len(named)-1)
    groups, n = np.unique(np.column_stack((*bins,quality,state)),axis=0,return_counts=True)
    return [dict(zip(CENSUS_FIELDS,(region,*map(int,row),int(count)))) for row,count in zip(groups,n)]


def prepare_work():
    inputs = {}
    result = {}
    for side in ('south','north'):
        _, t = working_index(side,inputs)
        keep = t['state']==4
        ids = t['ls_id'][keep]
        if len(np.unique(ids))!=len(ids):
            raise ValueError('Repeated selected native identity')
        result[side] = np.sort(ids)
    path = DEST/'working_ids.npz'
    path.parent.mkdir(exist_ok=True,parents=True)
    atomic_npz(path,**result)
    save_json(DEST/'working_ids.json',dict(inputs=inputs,outputs={path.name:sha(path)}))


def process(part, overrides, working):
    region = part['region']
    frozen = region=='south_v35'
    columns = ('TYPE','ls_id','source_id','RA','DEC','EBV')
    old = ('observed_r','dereddened_g_r','primary','band_state','nominal_numerical_flag',
           'qrf_quantiles_kpc','qrf_bins')
    new = ('r','r0','g0','z0','row_state','north_primary_region',*DEPTH_FIELDS)
    path = verified(ROOT/part['photo_file'],part['photo_sha256'])
    p = read_columns(path,columns+(old if frozen else new))
    owns,_ = owner_rows(part['side'],p,overrides,frozen=frozen)
    if len(owns)!=int(part['native_rows']) or owns.sum()!=int(part['owned_rows']):
        raise ValueError(f'Ownership changed in {path}')
    r = p['observed_r'] if frozen else p['r']
    color = p['dereddened_g_r'] if frozen else p['g0']-p['r0']
    lon,lat = galactic(p['RA'],p['DEC'])
    if frozen:
        state = np.full(len(r),3,'u1')
        quality = np.full(len(r),7,'u1')
        idx = np.arange(len(r))
        q,bins = p['qrf_quantiles_kpc'],p['qrf_bins']
    else:
        state_path = verified(ROOT/part['state_file'],part['state_sha256'])
        state_table = read_columns(state_path,('ls_id','TYPE','state'))
        np.testing.assert_array_equal(state_table['ls_id'],p['ls_id'])
        np.testing.assert_array_equal(state_table['TYPE'],p['TYPE'])
        state,quality = state_table['state'],p['row_state']
        idx = np.flatnonzero(state==3)
        q,bins = np.empty((0,5)),np.empty((0,6))
        if len(idx):
            pdf = read_columns(verified(ROOT/part['pdf_file'],part['pdf_sha256']),
                               ('photo_row','TYPE','ls_id','qrf_quantiles_kpc','qrf_bins'))
            np.testing.assert_array_equal(pdf['photo_row'],idx)
            np.testing.assert_array_equal(pdf['ls_id'],p['ls_id'][idx])
            np.testing.assert_array_equal(pdf['TYPE'],p['TYPE'][idx])
            q,bins = pdf['qrf_quantiles_kpc'],pdf['qrf_bins']
    if len(idx)!=int(part['native_pdf_rows']) or owns[idx].sum()!=int(part['owned_pdf_rows']):
        raise ValueError(f'PDF/state counts changed in {path}')
    if not np.all(np.isfinite(q)&(q>0)) or (bins<0).any():
        raise ValueError(f'Invalid PDF summaries in {path}')
    np.testing.assert_allclose(bins.sum(axis=1),1,rtol=0,atol=1e-7)
    hist = histogram(region,r[owns],color[owns],lat[owns],p['EBV'][owns],quality[owns],state[owns])
    science = np.zeros(len(idx),'u1')
    if frozen:
        near = (p['primary'] & ~p['nominal_numerical_flag'] & (r>=16) & (r<18)
                & (color>=.8) & (color<1.4) & np.isin(p['band_state'][:,4],(2,4)))
        science[near&(q[:,2]<2)] = 1
    elif region in ('north','south_low'):
        candidates = p['ls_id'][idx]
        keys = working[part['side']]
        j = np.searchsorted(keys,candidates)
        sel = np.zeros(len(idx),bool)
        inside = j<len(keys)
        sel[inside] = keys[j[inside]]==candidates[inside]
        if np.any(sel&~owns[idx]) or np.any(sel&(quality[idx]!=7)):
            raise ValueError(f'Working selection differs from the release in {path}')
        science[sel&(q[:,2]>=2)&(q[:,2]<5)] = 2
        science[sel&(q[:,2]>=5)&(q[:,2]<10)] = 3
        science[sel&(science==0)] = 4
    w = owns[idx] & (science>0)
    numbers = idx[w]
    t = None
    if len(numbers):
        t = {key:p[key][numbers] for key in columns}
        t.update(r=r[numbers], color=color[numbers], lon=lon[numbers], lat=lat[numbers],
                 q=q[w], bins=bins[w], science=science[w],
                 region=np.full(len(numbers),REGIONS.index(region),'u1'))
    return hist,t


def run(group):
    parts,overrides,inputs = indices()
    marker = DEST/'working_ids.json'
    work_receipt = json.loads(marker.read_text())
    name = 'working_ids.npz'
    working = read_columns(verified(DEST/name,work_receipt['outputs'][name]),('north','south'))
    inputs['working_ids'] = sha(marker)
    ix = np.array_split(np.arange(len(parts)),C['scan_shards'])[group]
    dest = DEST/'scan'
    dest.mkdir(exist_ok=True,parents=True)
    counts = {}
    selected = []
    with ThreadPoolExecutor(max_workers=int(os.getenv('SLURM_CPUS_PER_TASK','1'))) as pool:
        for hist,t in pool.map(lambda i:process(parts[i],overrides,working),ix):
            for row in hist:
                key = tuple(row[k] for k in CENSUS_FIELDS[:-1])
                counts[key] = counts.get(key,0)+row['n']
            if t is not None:
                selected.append(t)
    path = dest/f'{group:02d}.csv'
    csv_rows(path,[dict(zip(CENSUS_FIELDS,(*k,v))) for k,v in sorted(counts.items())])
    outputs = {path.name:sha(path)}
    if selected:
        t = join(selected)
        if len(np.unique(t['ls_id']))!=len(t['ls_id']):
            raise ValueError('Repeated owned PSF within a shard')
        path = dest/f'{group:02d}.npz'
        atomic_npz(path,**t)
        outputs[path.name] = sha(path)
    inputs['script'] = sha(ROOT/'scripts/final_science_catalog.py')
    save_json(dest/f'{group:02d}.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,outputs=outputs,
                  parts=len(ix),owned=sum(counts.values()),selected=sum(len(t['ls_id']) for t in selected)))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID','0')))
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare_work()
    else:
        run(args.shard)

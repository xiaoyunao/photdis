#!/usr/bin/env python3
"""Recover native-parent exclusion reasons for unmatched archived references."""
import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitsio
import numpy as np

from photdis.stage1 import legacy_ids, require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_reference_support import separation
from v3_5_dr11_area import tile_map
from v3_5_dr11_inventory import CONFIG, OUT as V35
from v3_5_dr11_pilot import LEDGER
from v3_5_dr11_production import FREEZE, PRODUCTION
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import join_parts
from v3_6_reference_index import OUT as INDEX, STATUSES, workers
from v3_6_vac_distances import ARCHIVE, DISTANCE_EDGES, read_columns


OUT = ROOT/'data/v3_6/native_losses'
ARCHIVES = {'v3_0':ROOT/'data/v3_0/references.npz','baseline_v1_4':ARCHIVE/'references.npz'}
UNMAPPED = len(LEDGER)
NOT_FOUND = UNMAPPED+1
POSITION = NOT_FOUND+1
DUPLICATE = POSITION+1
COUNTS = (*LEDGER,'outside_frozen_sweep_or_unusable_coordinate','unmatched_native_key',
          'native_position_disagreement','duplicate_native_key')


def sweep_owners(ra,dec,lookup):
    ok = np.isfinite(ra)&np.isfinite(dec)&(abs(dec)<=90)
    out = np.full(len(ra),-1,'i2')
    out[ok] = lookup[(ra[ok]%360).astype(int),np.minimum((dec[ok]+90).astype(int),179)]
    return out


def attach(path,inventory,marker,query):
    stat = path.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(inventory['size'],inventory['mtime_ns']):
        raise ValueError(f'Frozen native sweep changed: {path}')
    dest = PRODUCTION/path.stem
    rec = json.loads(verified(dest/'complete.json',marker).read_text())
    ledger = load(verified(dest/'parent_status.npz',rec['outputs']['parent_status.npz']))['status']
    with fitsio.FITS(str(path)) as f:
        hdu = f[1]
        if len(ledger)!=hdu.get_nrows():
            raise ValueError(f'Native row count differs from frozen ledger: {path}')
        native = hdu.read(columns=['RELEASE','BRICKID','OBJID'])
        ids = legacy_ids(native)
        order = np.argsort(ids,kind='stable')
        keys = ids[order]
        lo = np.searchsorted(keys,query['ls_id'],side='left')
        hi = np.searchsorted(keys,query['ls_id'],side='right')
        del ids,keys,native
        valid = hi-lo==1
        idx = np.full(len(lo),-1,'i8')
        idx[valid] = order[lo[valid]]
        status = np.where(hi-lo>1,DUPLICATE,NOT_FOUND).astype('i1')
        psf = np.full(len(idx),False)
        observed_r = np.full(len(idx),np.nan)
        color = np.full(len(idx),np.nan)
        if valid.any():
            use = np.flatnonzero(valid)
            rows,back = np.unique(idx[use],return_inverse=True)
            cols = ['RA','DEC','TYPE','FLUX_R','FLUX_G','MW_TRANSMISSION_G','MW_TRANSMISSION_R']
            values = hdu.read(columns=cols,rows=rows)[back]
            close = separation(query['RA'][use],query['DEC'][use],values['RA'],values['DEC'])
            status[use] = np.where(close,ledger[idx[use]],POSITION)
            psf[use] = close&(np.char.strip(values['TYPE'].astype('U'))=='PSF')
            with np.errstate(divide='ignore',invalid='ignore'):
                f_r = values['FLUX_R'].astype('f8')
                f_g = values['FLUX_G'].astype('f8')
                t_r,t_g = values['MW_TRANSMISSION_R'],values['MW_TRANSMISSION_G']
                r = np.where(f_r>0,22.5-2.5*np.log10(f_r),np.nan)
                gmr = np.where((f_g>0)&(f_r>0)&(t_g>0)&(t_r>0),-2.5*np.log10((f_g/t_g)/(f_r/t_r)),np.nan)
            observed_r[use[close]] = r[close]
            color[use[close]] = gmr[close]
    query.update(status=status,native_row=np.where(status<UNMAPPED,idx,-1),
                 native_PSF=psf,observed_r=observed_r,dereddened_g_r=color)
    return query


def inputs():
    f = DELIVERY/'complete.json'
    d = json.loads(f.read_text())
    freeze = json.loads(verified(FREEZE,d['production_freeze_sha256']).read_text())
    inv = V35/'inventory/complete.json'
    data = json.loads(verified(inv,freeze['inputs'][str(inv.relative_to(ROOT))]).read_text())
    with verified(DELIVERY/'sweeps.csv',d['outputs']['sweeps.csv']).open() as stream:
        checks = {r['sweep']:r['receipt_sha256'] for r in csv.DictReader(stream)}
    return freeze,data['sweeps'],checks


def shard(number,count):
    freeze,inventory,checks = inputs()
    all_names = {n for group in freeze['groups'] for n in group}
    work = {n for group in freeze['groups'][number::count] for n in group}
    tiles = [t for t in inventory if t['filename'] in all_names]
    lookup = tile_map(tiles)
    by_name = {t['filename']:i for i,t in enumerate(tiles)}
    relevant = np.array([by_name[n] for n in work])
    marker = INDEX/'complete.json'
    rec = json.loads(marker.read_text())
    queue = []
    for key,path in ARCHIVES.items():
        m = load(verified(INDEX/f'{key}.npz',rec['outputs'][f'{key}.npz']))
        raw = read_columns(verified(path,rec['inputs'][str(path.relative_to(ROOT))]),('ls_id','RA','DEC'))
        missing = m['match_status']==STATUSES.index('no_supported_match')
        owner = sweep_owners(raw['RA'],raw['DEC'],lookup)
        active = missing&np.isin(owner,relevant)
        if number==0:
            active |= missing&(owner<0)
        rows = np.flatnonzero(active)
        queue.append(dict(**{k:v[rows] for k,v in raw.items()},scope=np.full(len(rows),key),
                          reference_row=rows,owner=owner[rows]))
    queries = join_parts(queue)
    owner = queries.pop('owner')
    order = np.argsort(owner,kind='stable')
    breaks = np.r_[0,np.flatnonzero(np.diff(owner[order]))+1,len(order)]
    groups = [order[a:b] for a,b in zip(breaks[:-1],breaks[1:]) if b>a]
    root = Path(json.loads(CONFIG.read_text())['sweep_root'])

    def process(j):
        q = {k:v[j] for k,v in queries.items()}
        index = owner[j[0]]
        if index>=0:
            tile = tiles[index]
            name = tile['filename']
            return attach(root/name,tile,checks[name],q)
        q.update(status=np.full(len(j),UNMAPPED,'i1'),native_row=np.full(len(j),-1,'i8'),
                 native_PSF=np.zeros(len(j),bool),observed_r=np.full(len(j),np.nan),
                 dereddened_g_r=np.full(len(j),np.nan))
        return q

    with ThreadPoolExecutor(max_workers=workers()) as pool:
        data = join_parts(list(pool.map(process,groups)))
    dest = OUT/f'shard_{number:02d}'
    dest.mkdir(parents=True,exist_ok=True)
    part = dest/'losses.npz'
    atomic_npz(part,**data)
    save_json(dest/'complete.json',dict(shard=number,n_shards=count,job=os.getenv('SLURM_JOB_ID'),
        reference_index_sha256=sha(marker),code_sha256=sha(Path(__file__)),output_sha256=sha(part),
        rows=len(owner),status_counts={k:int((data.get('status',np.empty(0))==i).sum())
                                      for i,k in enumerate(COUNTS)}))
    print(number,len(owner),flush=True)


def collect(count):
    rec = json.loads((INDEX/'complete.json').read_text())
    hashes = {}
    batches = []
    for i in range(count):
        dest = OUT/f'shard_{i:02d}'
        path = dest/'complete.json'
        q = json.loads(path.read_text())
        if (q['shard'],q['n_shards'],q['reference_index_sha256'],q['code_sha256'])!=(
                i,count,sha(INDEX/'complete.json'),sha(Path(__file__))):
            raise ValueError(f'Inconsistent native loss shard: {path}')
        hashes[str(path.relative_to(ROOT))] = sha(path)
        batch = load(verified(dest/'losses.npz',q['output_sha256']))
        if batch:
            batches.append(batch)
    a = join_parts(batches)
    rows = []
    for key,path in ARCHIVES.items():
        m = load(verified(INDEX/f'{key}.npz',rec['outputs'][f'{key}.npz']))
        p = read_columns(verified(path,rec['inputs'][str(path.relative_to(ROOT))]),
                         ('specdis_distance','spdist_distance','p_corr','gaia_parallax_error','parallax_eligible'))
        select = a['scope']==key
        j = a['reference_row'][select]
        expected = np.flatnonzero(m['match_status']==STATUSES.index('no_supported_match'))
        np.testing.assert_array_equal(np.sort(j),expected)
        strong = p['parallax_eligible']&(p['p_corr']>=10*p['gaia_parallax_error'])&(p['p_corr']>0)
        gaia = np.divide(1.,p['p_corr'],out=np.full(len(strong),np.nan),where=strong)
        state = a['status'][select]
        for source,d in [('all_missing_supported',None),('Gaia_geometry',gaia),
                         ('SpecDis',p['specdis_distance']),('SPDist',p['spdist_distance'])]:
            bins = [(None,None,np.ones(len(j),bool))] if d is None else [
                (lo,hi,(d[j]>=lo)&(d[j]<hi)) for lo,hi in zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])]
            for lo,hi,mask in bins:
                counts = {name:int((mask&(state==k)).sum()) for k,name in enumerate(COUNTS)}
                if sum(counts.values())!=int(mask.sum()):
                    raise ValueError('Unaccounted native exclusion status')
                rows.append(dict(scope=key,reference=source,distance_lo=lo,distance_hi=hi,
                                 total=int(mask.sum()),**counts))
    output = OUT/'native_losses.csv'
    csv_rows(output,rows)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),shards=hashes,
              interpreted_as='Exclusive native reasons for the historical-classified no-supported-key cohort.',
              outputs={output.name:sha(output)}))
    print(len(a['reference_row']),len(rows),flush=True)


def main():
    require_compute()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('shard','collect'))
    p.add_argument('--shards',type=int,default=8)
    p.add_argument('--index',type=int,default=0)
    a = p.parse_args()
    collect(a.shards) if a.mode=='collect' else shard(a.index,a.shards)


if __name__ == '__main__':
    main()

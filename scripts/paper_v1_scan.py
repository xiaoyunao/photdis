#!/usr/bin/env python3
"""Measure native observable support and recover exact PDFs for the paper benchmark."""
from collections import defaultdict
import json
import os
from pathlib import Path

import fitsio
import healpy as hp
import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_0_native import legacy_ids
from v3_3_sed import verified
from v3_5_dr11_inventory import CONFIG as SOUTH_CONFIG, OUT as SOUTH
from v3_6_vac_distances import read_columns
from v3_9_unified import owner_rows
from final_science_catalog import REGIONS, indices
from final_catalogue_benchmark import OUT as BENCHMARK, load_parent

OUT = ROOT/'data/paper_v1_evidence'
SHARDS = 24
AXES = ('r','g_r','r_z','max_grz_error','ebv','sky')
EDGES = ([-np.inf,16,18,19,20,21,22,23,np.inf],
         [-np.inf,.0,.35,.5,.65,.8,1.05,1.4,np.inf],
         [-np.inf,.15,.35,.65,np.inf],
         [-np.inf,.035,.085,.16,np.inf], [-np.inf,.015,.025,np.inf])
DIMS = (4, *(len(e) for e in EDGES), 48)
PDF_COLUMNS = ('qrf_mu_edges','qrf_mu_mass')
DETAIL = ('ls_id','TYPE','source_id','region','r','g0','r0','z0','grz_error','ebv','cell',
          'tier','qrf_mu_edges','qrf_mu_mass')


def input_table():
    folder = ROOT/'data/final_science/evidence'
    p = folder/'complete.json'
    a = json.loads(p.read_text())
    core = read_columns(verified(folder/'selected.npz',a['outputs']['selected.npz']),
                        ('TYPE','ls_id','science'))
    assert (core['TYPE']=='PSF').all()
    sel = core['science']<=2
    ids, tier = core['ls_id'][sel], core['science'][sel]
    k = np.argsort(ids)
    b, _ = load_parent()
    assert (b['TYPE']=='PSF').all()
    return ids[k],tier[k],np.sort(b['ls_id']),sha(p),sha(BENCHMARK/'parent.json')


def locate(keys, ids):
    k = np.searchsorted(keys,ids)
    ok = k<len(keys)
    ok[ok] = keys[k[ok]]==ids[ok]
    return np.flatnonzero(ok), k[ok]


def old_photometry(part_rows, sweep, cfg, inventory):
    record = inventory[sweep]
    p = Path(cfg['sweep_root'])/record['filename']
    s = p.stat()
    assert (s.st_size,s.st_mtime_ns)==(record['size'],record['mtime_ns'])
    cols = ['RELEASE','BRICKID','OBJID','TYPE',*(f'{a}_{b}' for a in ('FLUX','FLUX_IVAR','MW_TRANSMISSION') for b in 'GRZ')]
    source = np.concatenate([t['parent_row'] for t in part_rows])
    # Read the native row positions without reprocessing other sweep objects.
    order = np.argsort(source)
    inverse = np.argsort(order)
    with fitsio.FITS(p) as f:
        raw = f[1].read(columns=cols, rows=source[order])[inverse]
    np.testing.assert_array_equal(legacy_ids(raw),np.concatenate([x['ls_id'] for x in part_rows]))
    assert np.all(np.char.strip(raw['TYPE'].astype(str))=='PSF')
    flux,ivar,trans = (np.column_stack([raw[f'{a}_{b}'] for b in 'GRZ']).astype('f8')
                       for a in ('FLUX','FLUX_IVAR','MW_TRANSMISSION'))
    with np.errstate(invalid='ignore',divide='ignore'):
        mag = 22.5-2.5*np.log10(flux/trans)
        err = 2.5/np.log(10)/flux/np.sqrt(ivar)
    return np.split(np.column_stack((mag,err)),np.cumsum([len(t['ls_id']) for t in part_rows])[:-1])


def cells(region,t,m,e):
    values = (t['observed_r'],m[:,0]-m[:,1],m[:,1]-m[:,2],e.max(axis=1),t['EBV'])
    bins = [np.where(np.isfinite(x),np.searchsorted(ed,x,side='right')-1,len(ed)-1) for x,ed in zip(values,EDGES)]
    pix = hp.ang2pix(2,t['RA'],t['DEC'],lonlat=True,nest=True)
    return np.ravel_multi_index((np.full(len(pix),region),*bins,pix),DIMS)


def count_sparse(out,keys):
    ids,n = np.unique(keys,return_counts=True)
    out[ids] += n


def concatenate_reference(blocks, fields):
    if not blocks:
        return {}
    result = {}
    for k in fields:
        values = [t[k] for t in blocks]
        if k in PDF_COLUMNS:
            width=max(v.shape[1] for v in values)
            if k=='qrf_mu_edges':
                values=[np.concatenate((v,v[:,-1:] + np.arange(1,width-v.shape[1]+1)*.1),axis=1)
                        if v.shape[1]!=width else v for v in values]
            else:
                values=[np.pad(v,((0,0),(0,width-v.shape[1]))) if v.shape[1]!=width else v for v in values]
        result[k]=np.concatenate(values)
    return result


def scan(group):
    parts, overrides, inputs = indices()
    ids,tier,bench,inputs['core'],inputs['benchmark'] = input_table()
    old = json.loads(SOUTH_CONFIG.read_text())
    marker = SOUTH/'inventory/complete.json'
    inventory = {Path(x['filename']).stem:x for x in json.loads(marker.read_text())['sweeps']}
    inputs['old_inventory'] = sha(marker)
    out = np.zeros((3,int(np.prod(DIMS))),'i8')
    links, detail = [], []
    shard = np.array_split(np.arange(len(parts)), SHARDS)[group]
    by_sweep = defaultdict(list)
    for idx in shard:
        p = parts[idx]
        if p['pdf_file']:
            by_sweep[p['region'],p['sweep']].append(p)
        else:
            assert int(p['owned_pdf_rows'])==0
    columns = ('ls_id','TYPE','source_id','RA','DEC','EBV','observed_r','dereddened_g_r')
    for (domain,sweep), group_parts in by_sweep.items():
        region = REGIONS.index(domain)
        frozen = domain=='south_v35'
        extra = ('parent_row',) if frozen else ('north_primary_region','mag_grz','error_grz')
        photos = [read_columns(verified(ROOT/p['photo_file'],p['photo_sha256']),columns+extra) for p in group_parts]
        restore = old_photometry(photos,sweep,old,inventory) if frozen else [None]*len(photos)
        for p, t, z in zip(group_parts,photos,restore):
            owner, _ = owner_rows(p['side'],t,overrides,frozen=frozen)
            assert owner.sum()==int(p['owned_rows'])
            if frozen:
                source = np.arange(len(owner))
                m,e = z[:,:3],z[:,3:]
                np.testing.assert_allclose(m[:,0]-m[:,1],t['dereddened_g_r'],atol=2e-5,rtol=0)
            else:
                d = read_columns(verified(ROOT/p['pdf_file'],p['pdf_sha256']),('ls_id','TYPE','photo_row'))
                source = d['photo_row']
                np.testing.assert_array_equal(d['ls_id'],t['ls_id'][source])
                assert np.all(d['TYPE']=='PSF')
                m,e = t['mag_grz'],t['error_grz']
            native = np.flatnonzero(owner[source])
            i = source[native]
            assert len(i)==int(p['owned_pdf_rows'])
            cell = cells(region,t,m,e)
            count_sparse(out[0],cell[i])
            k,j = locate(ids,t['ls_id'][i])
            tiers = tier[j]
            for s in (1,2):
                count_sparse(out[s],cell[i[k[tiers==s]]])
            links.append(dict(ls_id=t['ls_id'][i[k]],cell=cell[i[k]],tier=tiers))
            k,_ = locate(bench,t['ls_id'][i])
            if len(k):
                u = i[k]
                with np.load(ROOT/p['pdf_file'],allow_pickle=False) as pdf:
                    curves = {f:pdf[f][native[k]] for f in PDF_COLUMNS}
                    np.testing.assert_array_equal(pdf['ls_id'][native[k]],t['ls_id'][u])
                detail.append(dict(ls_id=t['ls_id'][u],TYPE=t['TYPE'][u],source_id=t['source_id'][u],
                    region=np.full(len(u),region,'u1'),r=t['observed_r'][u],g0=m[u,0],r0=m[u,1],
                    z0=m[u,2],grz_error=e[u],ebv=t['EBV'][u],cell=cell[u],
                    tier=np.zeros(len(u),'u1'),**curves))
    folder = OUT/'scan'
    folder.mkdir(exist_ok=True,parents=True)
    paths = [folder/f'{group:02}_{x}.npz' for x in ('counts','links','reference')]
    atomic_npz(paths[0],counts=out)
    for path,blocks,fields in zip(paths[1:],(links,detail),(('ls_id','cell','tier'),DETAIL)):
        atomic_npz(path,**concatenate_reference(blocks,fields))
    save_json(folder/f'{group:02}.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
             partitions=len(shard),counts=out.sum(axis=1).tolist(),reference=sum(len(t['ls_id']) for t in detail),
             outputs={p.name:sha(p) for p in paths}))


if __name__=='__main__':
    require_compute()
    scan(int(os.environ['SLURM_ARRAY_TASK_ID']))

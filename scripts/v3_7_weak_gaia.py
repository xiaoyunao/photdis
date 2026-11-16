#!/usr/bin/env python3
"""Score frozen QRF histograms for the full weak-Gaia 5–10 kpc assignment."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.special import log_ndtr, logsumexp

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import join_parts, validation_history
from v3_6_catalog_projection import OUT as PROJECTION
from v3_6_gaia_census import OUT as GAIA, STATES
from v3_6_reference_index import OUT as INDEX
from v3_6_vac_distances import read_columns
from v3_6_weak_gaia import BOUND, ODDS, OFFSETS, SCALES, histogram, native_qrf, score


OUT = ROOT/'data/v3_7/weak_gaia'
PROTOCOL = ROOT/'configs/v3_7_selection.json'
_STATE = None
POINTERS = ('sweep_number','part_number','part_row')


def sample(census):
    fields = ('catalog_row','TYPE','ls_id','identity','source_id','RA','DEC','observed_r',
              'dereddened_g_r','primary','strict','p_corr','gaia_parallax_error')
    weak = STATES.index('weak_signed_parallax')
    pieces, excluded = [], {'non_psf':0, 'unverified_type':0}
    for name, digest in census['child_receipts'].items():
        marker = verified(GAIA/name,digest)
        shard = json.loads(marker.read_text())
        payload = marker.with_suffix('.npz')
        with np.load(verified(payload,shard['outputs'][payload.name]),allow_pickle=False) as z:
            d = z['qrf_quantiles_kpc'][:,2]
            use = (z['gaia_state']==weak)&(d>=5)&(d<10)
            if 'TYPE' not in z:
                excluded['unverified_type'] += int(use.sum())
                continue
            kind = z['TYPE']
            excluded['non_psf'] += int((use&(kind!='PSF')&(kind!='')).sum())
            excluded['unverified_type'] += int((use&(kind=='')).sum())
            use &= kind=='PSF'
            if use.any():
                pieces.append({k:z[k][use] for k in fields})
    p = join_parts(pieces)
    if not p:
        raise ValueError('The frozen frame contains no eligible weak Gaia objects')
    key = p['catalog_row']
    if len(np.unique(key))!=len(key) or len(np.unique(p['ls_id']))!=len(key):
        raise ValueError('Repeat native identity in the Gaia sample')
    return p,excluded


def attach_pointers(p,projection):
    keys = p['catalog_row']
    found = np.zeros(len(keys),bool)
    for field in POINTERS:
        p[field] = np.full(len(keys),-1,dtype='i4')
    for path,digest in projection['child_receipts'].items():
        mark = verified(ROOT/path,digest)
        row = json.loads(mark.read_text())
        batch = mark.with_name('rows.npz')
        a = read_columns(verified(batch,row['outputs'][batch.name]),('catalog_row','ls_id','TYPE',*POINTERS))
        order = np.argsort(a['catalog_row'])
        ids = a['catalog_row'][order]
        if not len(ids):
            continue
        ix = np.flatnonzero((keys>=ids[0])&(keys<=ids[-1]))
        slots = np.searchsorted(ids,keys[ix])
        hit = ids[slots]==keys[ix]
        ix,loc = ix[hit],order[slots[hit]]
        np.testing.assert_array_equal(a['ls_id'][loc],p['ls_id'][ix])
        if not (a['TYPE'][loc]=='PSF').all() or found[ix].any():
            raise ValueError('Incorrect projection ownership')
        for field in POINTERS:
            p[field][ix] = a[field][loc]
        found[ix] = True
    if not found.all():
        raise ValueError('Missing native forecast pointers')


def one(i):
    p,pdfs,base = _STATE
    y,err = p['p_corr'][i],p['gaia_parallax_error'][i]
    q,w = histogram(*pdfs[i])
    live = np.isfinite(w)
    q,w = q[live],w[live]
    nominal = score(q,w,y,err)
    broad = score(*base,y,err)
    shift = (q<.1)[None,:]*np.log(ODDS)[:,None]
    trial = np.empty((len(OFFSETS),len(SCALES),len(ODDS)))
    for j,offset in enumerate(OFFSETS):
        for k,scale in enumerate(SCALES):
            z = scale*q+offset
            like = w-.5*((y-z)/err)**2-np.log(err*np.sqrt(2*np.pi))
            admit = w+log_ndtr(BOUND-z/err)
            trial[j,k] = logsumexp(like+shift,axis=1)-logsumexp(admit+shift,axis=1)
    return nominal,broad,trial


def block_interval(delta,block,draws,seed):
    """Pairs remain paired and a whole sky block is resampled together."""
    _,i = np.unique(block,return_inverse=True)
    sizes = np.bincount(i)
    sums = np.bincount(i,weights=delta)
    if len(sizes)<2:
        return (None,None)
    rng = np.random.default_rng(seed)
    resample = rng.integers(len(sizes),size=(draws,len(sizes)))
    return tuple(np.quantile(sums[resample].sum(axis=1)/sizes[resample].sum(axis=1),[.025,.975]))


def summarize(p,q,b,trial,protocol):
    r,c = p['observed_r'],p['dereddened_g_r']
    sky = hp.ang2pix(protocol['sky_nside'],p['RA'],p['DEC'],lonlat=True,nest=True)
    all_rows = np.ones(len(r),bool)
    rows,grid = [],[]
    for name,cut in (('supported',all_rows),('primary',p['primary']),('strict',p['strict'])):
        for label,group in (('all',all_rows),('faint',r>=18),('faint_blue',(r>=18)&(c>=.5)&(c<.8))):
            for history,keep in (('all',all_rows),('outside_recorded_history',~p['recorded_history'])):
                use = cut&group&keep
                if not use.any():
                    continue
                delta = q[use,0]-b[use,0]
                conf = block_interval(delta,sky[use],protocol['bootstrap_draws'],protocol['bootstrap_seed'])
                row = dict(selection=name,photometry=label,history=history,n=int(use.sum()),
                           negative=int((p['p_corr'][use]<0).sum()),blocks=len(np.unique(sky[use])))
                rows.append(dict(**row,observed_mas=p['p_corr'][use].mean(),predicted_mas=q[use,3].mean(),
                                 qrf_pit_mean=q[use,1].mean(),qrf_minus_broad=delta.mean(),lo=conf[0],hi=conf[1]))
                for j,offset in enumerate(OFFSETS):
                    for k,scale in enumerate(SCALES):
                        for l,odds in enumerate(ODDS):
                            change = trial[use,j,k,l]-q[use,0]
                            conf = block_interval(change,sky[use],protocol['bootstrap_draws'],protocol['bootstrap_seed'])
                            grid.append(dict(**row,offset_mas=offset,parallax_scale=scale,far_odds=odds,
                                             delta=change.mean(),lo=conf[0],hi=conf[1]))
    return rows,grid


def main():
    global _STATE
    require_compute()
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    paths = [PROTOCOL, GAIA/'complete.json', PROJECTION/'complete.json', DELIVERY/'complete.json',
             INDEX/'complete.json']
    inputs = {str(x.relative_to(ROOT)):sha(x) for x in paths}
    protocol,census,projection = (json.loads(x.read_text()) for x in paths[:3])
    p,excluded = sample(census)
    attach_pointers(p,projection)
    historical,legacy = validation_history(inputs,None)
    p['recorded_history'] = np.isin(p['source_id'],historical)|np.isin(p['ls_id'],legacy)|np.isin(p['identity'],historical)
    if (p['strict']&~p['primary']).any():
        raise ValueError('Strict sample is outside the frozen primary sample')
    print(f"Weak Gaia 5–10 kpc native PSFs: {len(p['ls_id']):,}",flush=True)
    pdfs = native_qrf(p,paths[3])
    base = histogram(np.array([5.,10+5*np.log10(20.)]),np.ones(1),order=64)
    _STATE = p,pdfs,base
    with ProcessPoolExecutor(max_workers=int(os.getenv('SLURM_CPUS_PER_TASK','1')),
                             mp_context=mp.get_context('fork')) as pool:
        q,b,grid = (np.stack(x) for x in zip(*pool.map(one,range(len(p['ls_id'])),chunksize=64)))
    np.testing.assert_allclose(grid[:,1,2,1],q[:,0],atol=1e-10,rtol=0)
    rows,stress = summarize(p,q,b,grid,protocol)
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_npz(OUT/'scores.npz',**p,qrf=q,broad=b,stress=grid)
    csv_rows(OUT/'scores.csv',rows)
    csv_rows(OUT/'stress.csv',stress)
    script = Path(__file__).resolve()
    inputs[str(script.relative_to(ROOT))] = sha(script)
    save_json(marker,dict(inputs=inputs,rows=len(q),excluded=excluded,
        outputs={name:sha(OUT/name) for name in ('scores.npz','scores.csv','stress.csv')}))


if __name__=='__main__':
    main()

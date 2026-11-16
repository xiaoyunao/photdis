#!/usr/bin/env python3
"""Evaluate the fixed candidate on the frozen weighted draw."""
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
from v3_6_gaia_census import OUT as GAIA, STATES
from v3_6_lf_sensitivity import OUT as LF
from v3_6_vac_distances import read_columns
from v3_6_weak_gaia import BOUND, gaia_rows, histogram, native_qrf, score
from v3_7_lf_odds import summarize
from v3_7_selection_replay import OUT as SELECTION, probabilities
from v3_7_weak_gaia import PROTOCOL

OUT = ROOT/'data/v3_8/candidate'
EDGES = np.r_[-np.inf,np.linspace(-.5,1.,61),np.inf]
_STATE = None


def read_cache(folder,name,receipt='complete.json'):
    marker = folder/receipt
    meta = json.loads(marker.read_text())
    return verified(folder/name,meta['outputs'][name])


def gates():
    marker = SELECTION/'complete.json'
    meta = json.loads(marker.read_text())
    prep = verified(SELECTION/'prepare.json',meta['inputs'][str((SELECTION/'prepare.json').relative_to(ROOT))])
    cfg = json.loads(prep.read_text())
    chunks = []
    fields = ('catalog_row','ls_id','TYPE','nominal_qnear')
    for i in range(cfg['shards']):
        name = f'replay_{i:02d}'
        child = SELECTION/f'{name}.json'
        verified(child,meta['inputs'][str(child.relative_to(ROOT))])
        chunks.append(read_columns(read_cache(SELECTION,name+'.npz',child.name),fields))
    records = join_parts(chunks)
    order = np.argsort(records['catalog_row'])
    if np.any(np.diff(records['catalog_row'][order])<=0):
        raise ValueError('Duplicate replay rows')
    if not np.all(records['TYPE']=='PSF'):
        raise ValueError('Expected native PSF replay')
    return {k:v[order] for k,v in records.items()}


def decisions(a,replay):
    q = probabilities(a,replay)
    score_values = a['dwarf_selection_score']
    if np.any(~np.isfinite(q[score_values>=.65])):
        raise ValueError('Incomplete candidate gate on its declared frame')
    np.testing.assert_array_equal(a['primary'],(score_values>=.80)&(q>=.90))
    return (score_values>=.65)&(q>=.90)


def conditional_mass(parallax,logw,error):
    threshold = log_ndtr(BOUND-parallax/error)
    # Work in log space even for high-significance true foreground.
    z = np.minimum(EDGES/error,BOUND)[:,None]-parallax[None,:]/error
    cdf = np.exp(logsumexp(log_ndtr(z)+logw,axis=1)-logsumexp(logw+threshold))
    return np.maximum(np.diff(cdf),0.)


def one(i):
    a,pdfs,base = _STATE
    e,y = a['gaia_parallax_error'][i],a['p_corr'][i]
    x,w = histogram(*pdfs[i])
    valid = np.isfinite(w)
    x,w = x[valid],w[valid]
    return score(x,w,y,e),score(*base,y,e),conditional_mass(x,w,e),conditional_mass(*base,e)


def weighted_interval(x,w,h,n,seed):
    _,ii = np.unique(h,return_inverse=True)
    groups = len(np.unique(ii))
    if groups<2:
        return [None,None]
    resampled = np.random.default_rng(seed).integers(groups,size=(n,groups))
    denom = np.bincount(ii,weights=w)[resampled].sum(axis=1)
    num = np.bincount(ii,weights=w*x)[resampled].sum(axis=1)
    return np.quantile(num/denom,[.025,.975]).tolist()


def forward(full,eligible,config,gaia,inputs):
    admission = gaia_rows(full,gaia)
    accepted = ((admission['gaia_state']==STATES.index('weak_signed_parallax'))&
                (full['qrf_quantiles_kpc'][:,2]>=5)&(full['qrf_quantiles_kpc'][:,2]<10)&eligible)
    p = {k:full[k][accepted] for k in ('ls_id','TYPE','sweep_number','part_number','part_row',
        'RA','DEC','source_id','identity','weight','primary','candidate')}
    p.update({k:admission[k][accepted] for k in ('p_corr','gaia_parallax_error')})
    ids,lsids = validation_history(inputs,None)
    p['outside_history'] = ~(np.isin(p['source_id'],ids)|np.isin(p['identity'],ids)|np.isin(p['ls_id'],lsids))
    global _STATE
    dpath = DELIVERY/'complete.json'
    base = histogram(np.array([5.,10+5*np.log10(20.)]),np.ones(1),order=64)
    _STATE = p,native_qrf(p,dpath),base
    with ProcessPoolExecutor(max_workers=min(16,int(os.getenv('SLURM_CPUS_PER_TASK','1'))),
                             mp_context=mp.get_context('fork')) as pool:
        q,b,qmass,bmass = map(np.stack,zip(*pool.map(one,range(len(p['ls_id'])),chunksize=16)))
    _STATE = None
    np.testing.assert_allclose(qmass.sum(axis=1),1.,atol=1e-11)
    np.testing.assert_allclose(bmass.sum(axis=1),1.,atol=1e-11)
    rows,curves = [],[]
    h = hp.ang2pix(config['sky_nside'],p['RA'],p['DEC'],nest=True,lonlat=True)
    for cut in ('primary','candidate'):
        for scope,m in (('all',p[cut]),('outside_recorded_history',p[cut]&p['outside_history'])):
            if not m.any():
                continue
            w = p['weight'][m]
            delta = q[m,0]-b[m,0]
            lo,hi = weighted_interval(delta,w,h[m],config['bootstrap_draws'],config['bootstrap_seed'])
            rows.append(dict(selection=cut,scope=scope,n=int(m.sum()),effective_n=w.sum()**2/(w@w),
                sky_blocks=len(np.unique(h[m])),negative=int((p['p_corr'][m]<0).sum()),
                observed=np.average(p['p_corr'][m],weights=w),predicted=np.average(q[m,3],weights=w),
                score=np.average(delta,weights=w),lo=lo,hi=hi))
            obs = np.histogram(p['p_corr'][m],EDGES,weights=w)[0]/w.sum()
            curves.extend(dict(selection=cut,scope=scope,lo=x,hi=z,observed=o,qrf=j,broad=k)
                for x,z,o,j,k in zip(EDGES[:-1],EDGES[1:],obs,
                    np.average(qmass[m],weights=w,axis=0),np.average(bmass[m],weights=w,axis=0)))
    return rows,curves


def main():
    require_compute()
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    OUT.mkdir(parents=True,exist_ok=True)
    draw = read_columns(read_cache(SELECTION,'draw.npz','prepare.json'),
        ('catalog_row','ls_id','TYPE','dwarf_selection_score','nominal_qnear','primary'))
    chosen = decisions(draw,gates())
    fields = ('TYPE','ls_id','catalog_row','identity','source_id','qrf_quantiles_kpc','qrf_bins','weight','RA','DEC',
              'primary','strict','scenario','class_shell','sweep_number','part_number','part_row')
    p = read_columns(read_cache(LF,'predictions.npz'),fields)
    order = np.argsort(draw['catalog_row'])
    pos = order[np.searchsorted(draw['catalog_row'][order],p['catalog_row'])]
    np.testing.assert_array_equal(draw['ls_id'][pos],p['ls_id'])
    np.testing.assert_array_equal(draw['primary'][pos],p['primary'])
    if not (p['TYPE']=='PSF').all() or not (draw['TYPE']=='PSF').all():
        raise ValueError('Both admitted inputs require native PSFs')
    p['candidate'] = chosen[pos]
    config = json.loads(PROTOCOL.read_text())
    candidate_lf = {**p,'primary':p['candidate']}
    diagnostics = (dict(r,selection='candidate',
        assignment='qrf_sed_cross_weight' if r['assignment']=='pdf_weighted' else 'median_bin')
        for r in summarize(candidate_lf,config) if r['selection']=='primary')
    csv_rows(OUT/'lf.csv',diagnostics)
    anchors = (SELECTION/'complete.json',GAIA/'complete.json',LF/'complete.json',
               DELIVERY/'complete.json',PROTOCOL,Path(__file__).resolve())
    inputs = {str(x.relative_to(ROOT)):sha(x) for x in anchors}
    weak,curves = forward(p,p['primary']|p['candidate'],config,json.loads(anchors[1].read_text()),inputs)
    csv_rows(OUT/'weak.csv',weak)
    csv_rows(OUT/'parallax_distribution.csv',curves)
    atomic_npz(OUT/'members.npz',**{k:p[k] for k in ('catalog_row','ls_id','TYPE','weight','primary','candidate')})
    save_json(marker,dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,rows=len(chosen),candidates=int(chosen.sum()),
        excluded_morphology={'non_psf':0,'unverified_type':0},
        scope='Frozen stratified draw with its original design weights and native QRF PDFs',
        outputs={x.name:sha(x) for x in OUT.iterdir() if x.is_file() and x!=marker}))


if __name__=='__main__':
    main()

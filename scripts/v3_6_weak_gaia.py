#!/usr/bin/env python3
"""Proper held-out signed-parallax scores on the stratified DR11 sample."""
import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_production import PRODUCTION
from v3_5_dr11_spatial import DELIVERY
from v3_5_dr11_validate import OUT as VALIDATION, take_rows, validation_history
from v3_6_gaia_census import OUT as GAIA
from v3_6_lf_sensitivity import NAMES as SED_MODELS, OUT as FORECAST
from v3_6_reference_index import OUT as INDEX
from v3_6_vac_distances import read_columns


OUT = ROOT/'data/v3_6/weak_gaia'
MODELS = ('QRF','uniform_mu',*SED_MODELS)
SCALES = np.array([.65,.8,1.,1.25,1.55])
OFFSETS = np.array([-.02,0.,.02])
ODDS = np.array([.25,1.,4.])
BOUND = 5.
_STATE = None


@lru_cache(maxsize=2)
def quadrature(order):
    return np.polynomial.legendre.leggauss(order)


def nodes(edges,order=12):
    x,w = quadrature(order)
    size = np.diff(edges)
    return (edges[:-1,None]+size[:,None]*(x+1)/2),size[:,None]*w/2


def histogram(edges,mass,order=12):
    fine = np.insert(edges,np.searchsorted(edges,15.),15.) if edges[0]<15.<edges[-1] and not (edges==15.).any() else edges
    ix = np.searchsorted(edges,.5*(fine[:-1]+fine[1:]))-1
    density = (mass/np.diff(edges))[ix]
    mu,w = nodes(fine,order)
    logw = np.full(w.shape,-np.inf)
    keep = density>0
    logw[keep] = np.log(w[keep]*density[keep,None])
    return 10**((10-mu.ravel())/5),logw.ravel()


def score(p,logw,y,err):
    """Condition on the actual weak-observation qualification y/error < 5."""
    upper = BOUND*err
    if not (np.isfinite(y) and np.isfinite(err) and err>0 and y<upper):
        raise ValueError('Signed observation is outside the weak-parallax test')
    z = (y-p)/err
    u = BOUND-p/err
    pass_log = log_ndtr(u)
    conditional = logsumexp(logw+pass_log)
    ll = logsumexp(logw-.5*z*z)-np.log(err*np.sqrt(2*np.pi))-conditional
    pit = np.exp(logsumexp(logw+log_ndtr(z))-conditional)
    negative = np.exp(logsumexp(logw+log_ndtr(-p/err))-conditional)
    w = np.exp(logw+pass_log-conditional)
    regular = u>=-12.
    lam = np.exp(-.5*u[regular]**2-.5*np.log(2*np.pi)-pass_log[regular])
    local_mean = np.empty_like(p)
    local_var = np.empty_like(p)
    local_mean[regular] = p[regular]-err*lam
    local_var[regular] = err**2*(1-u[regular]*lam-lam*lam)
    # Asymptotic inverse Mills ratio avoids cancellation far into the tail.
    t = -1/u[~regular]
    local_mean[~regular] = upper-err*t*(1-2*t**2+10*t**4-74*t**6)
    local_var[~regular] = err**2*t**2*(1-6*t**2+50*t**4)
    mean = w@local_mean
    sd = np.sqrt(w@(np.maximum(local_var,0)+(local_mean-mean)**2))
    return np.asarray([ll,min(pit,1.),min(negative,1.),mean,(y-mean)/sd if sd>0 else np.nan])


def one(i):
    s,pdfs,q,sed_grid,sed_nodes,sed_weights,base = _STATE
    y,err = q['p_corr'][i],q['gaia_parallax_error'][i]
    edges,mass = pdfs[i]
    pq,wq = histogram(edges,mass)
    values = [score(pq,wq,y,err),score(*base,y,err)]
    for curve in s['marginal_log_density'][i]:
        logw = np.interp(sed_nodes,sed_grid,curve)+sed_weights
        values.append(score(10**((10-sed_nodes)/5),logw,y,err))
    trial = np.empty((len(OFFSETS),len(SCALES),len(ODDS)))
    for a,offset in enumerate(OFFSETS):
        for b,scale in enumerate(SCALES):
            p = scale*pq+offset
            residual = (y-p)/err
            obs = wq-.5*residual**2-np.log(err*np.sqrt(2*np.pi))
            admitted = wq+log_ndtr(BOUND-p/err)
            # The far-odds stress changes the relative weight of the original
            # distant branch, before a shared parallax-scale stress is applied.
            far = pq<.1
            for c,factor in enumerate(ODDS):
                shift = np.where(far,np.log(factor),0.)
                trial[a,b,c] = logsumexp(obs+shift)-logsumexp(admitted+shift)
    return np.stack(values),trial


def gaia_rows(s,receipt):
    out = dict(gaia_state=np.full(len(s['ls_id']),-1,'i1'),
               p_corr=np.full(len(s['ls_id']),np.nan),gaia_parallax_error=np.full(len(s['ls_id']),np.nan))
    for name,digest in receipt['child_receipts'].items():
        marker = verified(GAIA/name,digest)
        shard = json.loads(marker.read_text())
        data = verified(marker.with_suffix('.npz'),shard['outputs'][marker.with_suffix('.npz').name])
        with np.load(data,allow_pickle=False) as z:
            r = z['catalog_row']
            order = np.argsort(r)
            slots = np.searchsorted(r[order],s['catalog_row'])
            hit = np.flatnonzero(slots<len(r))
            hit = hit[r[order[slots[hit]]]==s['catalog_row'][hit]]
            if not len(hit):
                continue
            match = order[slots[hit]]
            np.testing.assert_array_equal(z['ls_id'][match],s['ls_id'][hit])
            if not (z['TYPE'][match]=='PSF').all() or (out['gaia_state'][hit]>=0).any():
                raise ValueError('Weak sample lost native PSF or unique catalog ownership')
            for k in out:
                out[k][hit] = z[k][match]
    if (out['gaia_state']<0).any():
        raise ValueError('Gaia census does not cover the full frozen LF draw')
    return out


def native_qrf(s,source):
    receipt = json.loads(source.read_text())
    with verified(DELIVERY/'sweeps.csv',receipt['outputs']['sweeps.csv']).open() as f:
        hashes = {r['sweep']:r['receipt_sha256'] for r in csv.DictReader(f)}
    index = json.loads((INDEX/'complete.json').read_text())
    names = read_columns(verified(INDEX/'catalog_rows.npz',index['outputs']['catalog_rows.npz']),('sweep',))['sweep']
    jobs = np.unique(s['sweep_number'])
    def read(i):
        name = names[i]
        directory = PRODUCTION/Path(name).stem
        marker = json.loads(verified(directory/'complete.json',hashes[name]).read_text())
        parts = sorted(k for k in marker['outputs'] if k.startswith('part_'))
        use = np.flatnonzero(s['sweep_number']==i)
        result = {}
        for j in np.unique(s['part_number'][use]):
            slots = use[s['part_number'][use]==j]
            pos = s['part_row'][slots]
            p = parts[j]
            with np.load(verified(directory/p,marker['outputs'][p]),allow_pickle=False) as z:
                np.testing.assert_array_equal(s['ls_id'][slots],z['ls_id'][pos])
                assert (z['TYPE'][pos]=='PSF').all()
                result.update(zip(slots,zip(z['qrf_mu_edges'][pos],z['qrf_mu_mass'][pos])))
        return result
    found = {}
    with ThreadPoolExecutor(max_workers=min(12,int(os.getenv('SLURM_CPUS_PER_TASK','1')))) as pool:
        for rows in pool.map(read,jobs):
            found.update(rows)
    return [found[i] for i in range(len(s['ls_id']))]


def groups(s,old):
    r,c = s['observed_r'],s['dereddened_g_r']
    bright = (r>=14)&(r<18)
    faint = (r>=18)&(r<20)
    blue = (c>=.5)&(c<.8)
    d = s['qrf_quantiles_kpc'][:,2]
    all_rows = np.ones(len(r),bool)
    for cut,mask in (('supported',all_rows),('primary',s['primary']),('strict',s['strict'])):
        for history,seen in (('all',all_rows),('outside_recorded_history',~old)):
            for label,group in (('all',all_rows),('r14_18',bright),('r18_20',faint),
                                ('r18_20_blue',faint&blue),('r18_20_other_color',faint&~blue),
                                ('pred_2_5',(d>=2)&(d<5)),('pred_5_10',(d>=5)&(d<10)),
                                ('r18_20_pred_2_5',faint&(d>=2)&(d<5)),
                                ('r18_20_pred_5_10',faint&(d>=5)&(d<10))):
                yield (cut,history,label),mask&seen&group


def summaries(s,q,old,stats,trial):
    rows,pit,stress = [],[],[]
    for labels,use in groups(s,old):
        if not use.any():
            rows.append(dict(zip(('selection','history','group'),labels),n=0))
            continue
        w = s['weight'][use]
        common = dict(zip(('selection','history','group'),labels),n=len(w),weighted_n=w.sum(),
                      effective_n=w.sum()**2/(w@w),negative_n=int((q['p_corr'][use]<0).sum()),
                      observed_negative_fraction=np.average(q['p_corr'][use]<0,weights=w))
        for j,name in enumerate(MODELS):
            score_values = stats[use,j]
            rows.append(dict(**common,model=name,log_score=np.average(score_values[:,0],weights=w),
                relative_to_uniform_mu=np.average(score_values[:,0]-stats[use,1,0],weights=w),
                predicted_negative_fraction=np.average(score_values[:,2],weights=w),
                observed_mean_mas=np.average(q['p_corr'][use],weights=w),
                predicted_mean_mas=np.average(score_values[:,3],weights=w),
                standardized_residual_median=np.nanmedian(score_values[:,4])))
            bins = np.linspace(0.,1.,11)
            c,_ = np.histogram(score_values[:,1],bins=bins)
            wc,_ = np.histogram(score_values[:,1],bins=bins,weights=w)
            pit.extend(dict(**common,model=name,lo=a,hi=b,count=n,weighted_fraction=v/w.sum())
                       for a,b,n,v in zip(bins[:-1],bins[1:],c,wc))
        ll = np.average(trial[use],weights=w,axis=0)
        for i,offset in enumerate(OFFSETS):
            for j,scale in enumerate(SCALES):
                for k,odds in enumerate(ODDS):
                    stress.append(dict(**common,parallax_offset_mas=offset,parallax_scale=scale,
                        original_far_branch_odds_multiplier=odds,log_score=ll[i,j,k],
                        relative_to_frozen_qrf=ll[i,j,k]-ll[1,2,1]))
    return rows,pit,stress


def main():
    global _STATE
    require_compute()
    markers = [FORECAST/'complete.json',GAIA/'complete.json',DELIVERY/'complete.json',VALIDATION/'complete.json']
    raw,gaia,_,original = [json.loads(p.read_text()) for p in markers]
    p = verified(FORECAST/'predictions.npz',raw['outputs']['predictions.npz'])
    keys = ('catalog_row','ls_id','identity','source_id','TYPE','weight','primary','strict','observed_r',
            'dereddened_g_r','sweep_number','part_number','part_row','scenario','qrf_bins','qrf_quantiles_kpc')
    s = read_columns(p,keys)
    if not (s['TYPE']=='PSF').all() or list(s.pop('scenario'))!=list(SED_MODELS):
        raise ValueError('Expected the native LF draw and its frozen scenario list')
    q = gaia_rows(s,gaia)
    weak = q['gaia_state']==4
    s = take_rows(s,weak)
    q = take_rows(q,weak)
    with np.load(p,allow_pickle=False) as z:
        grid = z['grid_mu']
        s['marginal_log_density'] = z['marginal_log_density'][weak]
    hashes = {str(x.relative_to(ROOT)):sha(x) for x in (*markers,Path(__file__))}
    historical,legacy = validation_history(hashes,None)
    initial = verified(VALIDATION/'gaia_qualification_roster.npz',
                       original['outputs']['gaia_qualification_roster.npz'])
    seen = read_columns(initial,('ls_id','native_source_id'))
    old = (np.isin(s['source_id'],historical)|np.isin(s['ls_id'],legacy)
           |np.isin(s['source_id'],seen['native_source_id'])|np.isin(s['ls_id'],seen['ls_id']))
    print(f'Weak signed parallax sample: {len(old)}, outside recorded history: {(~old).sum()}',flush=True)
    pdfs = native_qrf(s,markers[2])
    x,w = nodes(grid,order=6)
    broad = histogram(np.array([5.,10+5*np.log10(20.)]),np.ones(1),order=64)
    _STATE = s,pdfs,q,grid,x.ravel(),np.log(w.ravel()),broad
    with ProcessPoolExecutor(max_workers=int(os.getenv('SLURM_CPUS_PER_TASK','16')),
                             mp_context=mp.get_context('fork')) as pool:
        stats,trial = [np.stack(x) for x in zip(*pool.map(one,range(len(old)),chunksize=24))]
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    names = ('scores.csv','pit.csv','forward_grid.csv')
    for name,rows in zip(names,summaries(s,q,old,stats,trial)):
        csv_rows(OUT/name,rows)
    keys = ('ls_id','source_id','weight','primary','strict','observed_r','dereddened_g_r','sweep_number')
    atomic_npz(OUT/'objects.npz',**{k:s[k] for k in keys},**q,recorded_history=old,
               stats=stats,stress_grid=trial,parallax_scales=SCALES,offsets_mas=OFFSETS,far_odds=ODDS)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),models=MODELS,n=len(old),
        negative=int((q['p_corr']<0).sum()),outside_history=int((~old).sum()),inputs=hashes,
        score_condition='Conditional on corrected signed S/N < 5. No Gaia posterior update.',
        stress_scope='Diagnostic parallax scale, far branch odds and common zero-point shifts. Uses the same sample for comparing scores, without an independent fitted adoption test.',
        outputs={n:sha(OUT/n) for n in (*names,'objects.npz')}))


if __name__=='__main__':
    main()

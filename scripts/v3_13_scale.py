#!/usr/bin/env python3
"""Swap distance teachers on identical references and measure soft-weight reliability."""
import argparse
import os

import numpy as np
from scipy.stats import norm

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_10_selection import ROUTES, references
from v3_11_layers import read_receipt
from v3_11_selection import DEST as SCORES
from v3_12_reference import DISTANCE, Z, M30, predict, redrock_and_physics, relation, separation

DEST = ROOT/'data/v3_13/scale'
R = (-np.inf,18,19,20,21,22,23)
COLOR = (-np.inf,.2,.35,.5,.8,1.4,np.inf)
FEH = (-np.inf,-1.5,-.8,-.3,np.inf)


def bins(values,edges):
    return [(f'{a:g}:{b:g}',(values>=a)&(values<b)) for a,b in zip(edges[:-1],edges[1:])]


def interval_bin(n,count):
    z = norm.ppf(.975)
    p = count/n
    m = (p+.5*z*z/n)/(1+z*z/n)
    h = z*np.sqrt(p*(1-p)/n+.25*z*z/n**2)/(1+z*z/n)
    return m-h,m+h


def reliability(t,p,good):
    color = t['g0']-t['r0']
    labels = p['labels']
    qualified = (t['qualified']=='QSO') | (t['qualified']=='GALAXY')
    known = good&((labels<3)|qualified)&np.isfinite(p['score_candidate'])
    ref = np.where((t['gaia_state']==6)&(t['reference_kpc']>0),t['reference_kpc'],
                   np.where(t['specdis_quality'],t['specdis_kpc'],np.nan))
    groups = [('all','all',np.ones(len(ref),bool))]
    for name,val,edges in (('gr',color,COLOR),('r',t['r'],R),('distance',ref,DISTANCE)):
        groups += [(name,k,v) for k,v in bins(val,edges)]
    rows = []
    s_edges = np.linspace(0.,1.00000001,11)
    for axis,key,frame in groups:
        for curve,score in (('original',p['score_original']),('revised',p['score_candidate'])):
            for name,classcut in [('known_classes',(labels<3)),('including_qualified_nonstellar',np.ones(len(ref),bool))]:
                for a,b in zip(s_edges[:-1],s_edges[1:]):
                    k = known&frame&classcut&(score>=a)&(score<b)
                    n = int(k.sum())
                    if n==0:
                        continue
                    nd = int((k&(labels==0)&~qualified).sum())
                    l,h = interval_bin(n,nd)
                    rows.append(dict(axis=axis,interval=key,score=curve,reference_classes=name,
                      score_lo=a,score_hi=b,n=n,dwarf=nd,giant=int((k&(labels==1)).sum()),
                      intermediate=int((k&(labels==2)).sum()),qualified_nonstellar=int((k&qualified).sum()),
                      mean_score=float(score[k].mean()),dwarf_rate=nd/n,wilson_lo=l,wilson_hi=h,
                      brier=float(np.mean((score[k]-((labels[k]==0)&~qualified[k]))**2))))
    return rows


def table(t,p,scope,forecasts,gaia,spec,spec_valid):
    col = t['g0']-t['r0']
    estimands = [('all','all',np.ones(len(col),bool))]
    for axis,v,edges in (('gr',col,COLOR),('r',t['r'],R),('feh',t['rvs_feh'],FEH)):
        estimands += [(axis,key,mask) for key,mask in bins(v,edges)]
    common = np.logical_and.reduce([np.isfinite(y).all(axis=1) for y in forecasts.values()])
    rows = []
    for name,r,usable in [('Gaia',gaia,np.isfinite(gaia)),('SpecDis',spec,spec_valid),
                          ('SPDist',t['spdist_kpc'],t['spdist_quality'])]:
        ok = usable&(r>0)
        for axis,key,mask in estimands:
            for label in (-2,0,1,2):
                cls = np.ones(len(col),bool) if label==-2 else p['labels']==label
                for lo,hi in zip(DISTANCE[:-1],DISTANCE[1:]):
                    frame = scope&ok&mask&cls&(r>=lo)&(r<hi)
                    if not frame.any():
                        continue
                    j = frame&common
                    mu_ref = 10+5*np.log10(r[j])
                    for method,q in forecasts.items():
                        rows.append(dict(reference=name,axis=axis,interval=key,label=label,d_lo=lo,d_hi=hi,
                                         method=method,available=int(frame.sum()),**distance_summary(q[j],mu_ref)))
    return rows


def run(side):
    t,inputs = references(side)
    p = read_receipt(SCORES/side,'complete.json','scores.npz',
        ('TYPE','ls_id','labels','new_train','new_cal','held_out','score_original','score_candidate'),inputs)
    np.testing.assert_array_equal(t['TYPE'],p['TYPE'])
    np.testing.assert_array_equal(t['ls_id'],p['ls_id'])
    redrock_and_physics(side,t,inputs)
    same = (separation(t['RA'],t['DEC'],M30)>=2.) & (t['source_id']>0)
    col = t['g0']-t['r0']
    r0 = t['r0']
    gaia = np.where((t['gaia_state']==6)&(t['reference_kpc']>0),t['reference_kpc'],np.nan)
    spec = t['specdis_kpc']
    spec_valid = t['specdis_quality']&np.isfinite(spec)&(spec>0)
    finite = same&np.isfinite(col)&np.isfinite(r0)
    tr = p['new_train']&finite&(p['labels']==0)
    g,s = tr&np.isfinite(gaia),tr&spec_valid
    dual = g&s
    mix = np.where(np.isfinite(gaia),gaia,spec)
    variants = [('Gaia paired',dual,gaia),('SpecDis paired',dual,spec),
                ('Gaia eligible',g,gaia),('SpecDis eligible',s,spec),
                ('Gaia first mixed',g|s,mix)]
    forecasts,models,fit_rows = {},{},[]
    for label,k,r in variants:
        mod = relation(col[k],r0[k]-10-5*np.log10(r[k]))
        mu,width = predict(mod,col,r0)
        forecasts[label] = mu[:,None]+width[:,None]*Z
        models[label] = mod
        fit_rows.extend(dict(method=label,gr=cc,Mr=aa,width=ww,n=int(nn))
              for cc,aa,ww,nn in zip(mod['color'],mod['absolute'],mod['width'],mod['n']))
    ev = (t['sky_role']==4)&t['evaluation_row']&finite
    paired_cal = p['new_cal']&finite&np.isfinite(gaia)&spec_valid&(p['labels']==0)
    paired_field = ev&np.isfinite(gaia)&spec_valid
    pairs = []
    for group,phase in [('train',dual),('calibration',paired_cal),('held_out',paired_field)]:
        for axis,v,edges in (('gr',col,COLOR),('r',t['r'],R),('feh',t['rvs_feh'],FEH),('Gaia kpc',gaia,DISTANCE)):
            for key,mask in bins(v,edges):
                j = phase&mask
                if j.any():
                    val = 5*np.log10(spec[j]/gaia[j])
                    pairs.append(dict(group=group,axis=axis,interval=key,n=int(j.sum()),
                            spec_minus_gaia_mu=np.median(val),mu_nmad=1.4826*np.median(abs(val-np.median(val)))))
    stats = table(t,p,ev,forecasts,gaia,spec,spec_valid)
    weights = reliability(t,p,ev)
    out = DEST/side
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('relations',fit_rows),('teacher_pairs',pairs),('distance',stats),('soft_reliability',weights)]:
        csv_rows(out/f'{name}.csv',rows)
    atomic_npz(out/'identity.npz',TYPE=t['TYPE'],ls_id=t['ls_id'],source_id=t['source_id'],
               dual_fit=dual,dual_cal=paired_cal,eval_role=ev,dual_eval=paired_field)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        fit={name:int(k.sum()) for name,k,_ in variants},test=int(ev.sum()),
        heldout_paired=int(paired_field.sum()),
        outputs={f.name:sha(f) for f in out.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('side',choices=tuple(ROUTES))
    run(parser.parse_args().side)

#!/usr/bin/env python3
"""Compare native color–luminosity distances and selection on fixed reference stars."""

import argparse
import json
import os

import joblib
import numpy as np
from scipy.special import ndtri

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_9_references import subset
from v3_10_science import working_index
from v3_10_selection import ROUTES, features, fit, references
from v3_10_working_sample import DOMAINS, RADIAL_MASS, SCORE
from v3_11_layers import evidence, read_receipt
from v3_11_selection import DEST as V311, matched_threshold

DEST = ROOT/'data/v3_12/reference'
COLOR = np.arange(-.15,2.51,.05)
DISTANCE = (0.,2.,5.,10.,np.inf)
Z = ndtri([.025,.16,.50,.84,.975])
M30 = (325.092,-23.180)


def separation(ra, dec, center):
    a,b = np.deg2rad([ra,dec])
    c,d = np.deg2rad(center)
    h = np.sin((b-d)/2)**2+np.cos(b)*np.cos(d)*np.sin((a-c)/2)**2
    return np.rad2deg(2*np.arcsin(np.sqrt(np.clip(h,0.,1.))))


def relation(color, absolute):
    centers, medians, widths, counts = [],[],[],[]
    for lo,hi in zip(COLOR[:-1],COLOR[1:]):
        k = (color>=lo)&(color<hi)&np.isfinite(absolute)
        if k.sum()>=30:
            m = np.median(absolute[k])
            centers.append((lo+hi)/2)
            medians.append(m)
            widths.append(max(.12,1.4826*np.median(abs(absolute[k]-m))))
            counts.append(int(k.sum()))
    if len(centers)<5:
        raise ValueError('Insufficient training support for the native main sequence')
    return dict(color=np.array(centers),absolute=np.array(medians),width=np.array(widths),n=np.array(counts))


def predict(model, color, r0):
    """Interpolate within the fitted color range, keeping unsupported holes explicit."""
    nodes = model['color']
    near = np.searchsorted(nodes,color)
    lo,hi = np.clip(near-1,0,len(nodes)-1),np.clip(near,0,len(nodes)-1)
    good = (color>=nodes[0])&(color<=nodes[-1])&(nodes[hi]-nodes[lo]<=.16)
    mu = np.where(good,r0-np.interp(color,nodes,model['absolute']),np.nan)
    width = np.where(good,np.interp(color,nodes,model['width']),np.nan)
    return mu,width


def redrock_and_physics(side,t,inputs):
    from v3_10_evidence import DEST as EVIDENCE
    for k in ('RA','DEC','g0','r0','rvs_feh','rvs_logg','rvs_teff'):
        t[k] = np.full(len(t['ls_id']),np.nan)
    for region in ROUTES[side]:
        cols = ('TYPE','ls_id','RA','DEC','g0','r0','rvs_feh','rvs_logg','rvs_teff')
        p = read_receipt(EVIDENCE/region,'complete.json','references.npz',cols,inputs)
        j = unique_join(t['ls_id'],p['ls_id'])
        k = j>=0
        if not np.all(p['TYPE'][j[k]]=='PSF'):
            raise ValueError('Unverified physics reference')
        for field in cols[2:]:
            t[field][k] = p[field][j[k]]
    # The bright-domain evidence includes both original and supplemental qualified Redrock.
    e = evidence(side,inputs)
    j = unique_join(t['ls_id'],e['ls_id'])
    t['qualified'] = np.full(len(j),'',dtype='U8')
    t['qualified'][j>=0] = e['nonstellar'][j[j>=0]]


def retention(w):
    return int(np.count_nonzero(w)),float(w.sum()),float(w.mean()) if len(w) else np.nan


def weighted_summary(residual, weight):
    good = np.isfinite(residual)&(weight>0)&np.isfinite(weight)
    y,w = residual[good],weight[good]
    if not len(y):
        return {}
    def med(v):
        j = np.argsort(v)
        return float(v[j][np.searchsorted(np.cumsum(w[j]),w.sum()/2)])
    relative = np.expm1(np.log(10)/5*y)
    m = med(relative)
    return dict(weighted_bias=m,weighted_nmad=1.4826*med(abs(relative-m)),
                weighted_severe=float(np.average(abs(y)>1,weights=w)))


def tables(t,labels,keep,q,simple,weights):
    results,migration,contaminants = [],[],[]
    convert = lambda d: 10+5*np.log10(d)
    gr = t['g0']-t['r0']
    phys = t['rvs_logg']
    groups = {
        'all_dwarf':labels==0,
        'blue_dwarf':(labels==0)&(gr<.5),
        'blue_turnoff_proxy':(gr<.5)&(labels==2)&(phys>=3.7)&(t['rvs_teff']>=5500),
        'cold_red_dwarf':(labels==0)&(gr>=.8),
        'intermediate':labels==2,
        'giant':labels==1,
    }
    for label, cohort in groups.items():
        for name,r,good in (('Gaia',t['reference_kpc'],t['gaia_state']==6),
                             ('SpecDis',t['specdis_kpc'],t['specdis_quality']),
                             ('SPDist',t['spdist_kpc'],t['spdist_quality'])):
            base = cohort&keep&good&np.isfinite(r)&(r>0)
            if not base.any():
                continue
            axes = [('all','all',np.ones(len(gr),bool))]
            for axis,val,edges in (('r',t['r'],[-np.inf,17.,18.,19.,20.]),
                                   ('gr',gr,[-np.inf,.2,.5,.8,1.4,np.inf]),
                                   ('feh',t['rvs_feh'],[-np.inf,-1.5,-.8,-.3,np.inf]),
                                   ('logg',phys,[-np.inf,3.2,3.7,4.2,4.6,np.inf])):
                axes.extend((axis,f'{a:g}:{b:g}',(val>=a)&(val<b)) for a,b in zip(edges[:-1],edges[1:]))
                if axis in ('feh','logg'):
                    axes.append((axis,'unknown',~np.isfinite(val)))
            for axis,interval,stratum in axes:
                for lo,hi in zip(DISTANCE[:-1],DISTANCE[1:]):
                    ix = np.flatnonzero(base&stratum&(r>=lo)&(r<hi))
                    if not len(ix):
                        continue
                    ref = convert(r[ix])
                    same = np.isfinite(simple[ix]).all(axis=1)
                    comparison = [('color_relation',simple[ix],same.astype(float)),
                                  ('qrf_common_color_support',q[ix],same.astype(float))]
                    comparison += [(key,q[ix],w[ix]) for key,w in weights.items()]
                    for method,p,w in comparison:
                        admitted = w>0
                        n,e,f = retention(w)
                        meta = dict(group=label,reference=name,axis=axis,interval=interval,
                                    d_lo=lo,d_hi=hi,method=method)
                        results.append(dict(**meta,available=len(ix),selected=n,effective_n=e,
                            effective_retention=f,color_relation_supported=int(same.sum()),
                            **distance_summary(p[admitted],ref[admitted]),
                            **weighted_summary(p[:,2]-ref,w)))
                        if axis=='all':
                            d_hat = 10**((p[:,2]-10)/5)
                            for a,b in zip(DISTANCE[:-1],DISTANCE[1:]):
                                hit = admitted&(d_hat>=a)&(d_hat<b)
                                migration.append(dict(**meta,pred_lo=a,pred_hi=b,available=len(ix),
                                    count=int(hit.sum()),effective=float(w[hit].sum())))
    for group,cls in (('giant',labels==1),('intermediate',labels==2),
                      ('QSO',t['qualified']=='QSO'),('GALAXY',t['qualified']=='GALAXY')):
        k = keep&cls
        for method,w in weights.items():
            n,e,f = retention(w[k])
            contaminants.append(dict(group=group,method=method,available=int(k.sum()),
                                    selected=n,effective_n=e,effective_retention=f))
    return results,migration,contaminants


def run(side):
    t,inputs = references(side)
    p = read_receipt(V311/side,'complete.json','scores.npz',
        ('TYPE','ls_id','labels','new_train','new_cal','held_out','radial_pass','score_original','score_candidate'),inputs)
    np.testing.assert_array_equal(t['ls_id'],p['ls_id'])
    np.testing.assert_array_equal(t['TYPE'],p['TYPE'])
    redrock_and_physics(side,t,inputs)
    field = separation(t['RA'],t['DEC'],M30)<2.
    gr = t['g0']-t['r0']
    good_ref = (t['gaia_state']==6)&(t['reference_kpc']>0)
    ref = np.where(good_ref,t['reference_kpc'],np.where(t['specdis_quality'],t['specdis_kpc'],np.nan))
    train = p['new_train']&~field&np.isfinite(gr)&np.isfinite(t['r0'])&np.isfinite(ref)&(ref>0)&(p['labels']==0)
    rel = relation(gr[train],t['r0'][train]-10-5*np.log10(ref[train]))
    mu,width = predict(rel,gr,t['r0'])
    q = t['qrf_quantiles_kpc']
    with np.errstate(invalid='ignore',divide='ignore'):
        q = 10+5*np.log10(q)
    selected = p['held_out']&~field&np.isfinite(q).all(axis=1)
    radial = p['radial_pass']
    old,new = p['score_original'],p['score_candidate']
    thresh = json.loads((V311/side/'complete.json').read_text())['threshold']
    _,catalog = working_index(side,inputs)
    j = unique_join(t['ls_id'],catalog['ls_id'])
    supported = j>=0
    supported[supported] &= catalog['state'][j[supported]]>=2
    cal = p['new_cal']&supported&~field&(t['region']==DOMAINS[side])
    cal_radial = np.zeros(len(j),bool)
    cal_radial[supported] = catalog['radial_mass'][j[supported]].sum(axis=1)>=RADIAL_MASS
    soft_limits = [1.]
    for label in (1,2):
        k = cal&(p['labels']==label)
        if k.any() and new[k].sum()>0:
            soft_limits.append(np.count_nonzero((old[k]>=SCORE)&cal_radial[k])/new[k].sum())
    scale = min(soft_limits)
    weights = dict(qrf_all=np.ones(len(q)),original_hard=(old>=SCORE)&radial,
                   revised_hard=(new>=thresh)&radial,
                   qrf_soft_dwarf=new,qrf_soft_matched=new*scale)
    x = features(side,t,inputs)
    label = np.where(p['labels']<3,p['labels'],-1)
    label[field] = -1
    reduced = {'full_feature_control':[], 'without_r':[2] if side=='north' else [0],
               'without_sky':[] if side=='north' else [1,2,3],
               'without_r_and_sky':[2] if side=='north' else [0,1,2,3]}
    models = {}
    for name,drop in reduced.items():
        if name=='without_sky' and side=='north':
            continue
        if name=='without_r_and_sky' and side=='north':
            continue
        use = np.delete(x,drop,axis=1)
        model,score,ti,ci,_ = fit(side,use,dict(t,class_label=label),inputs)
        cut,budgets = matched_threshold(score,old,p['labels'],ci&cal)
        models[name] = dict(model,threshold=cut,drop=drop,budgets=budgets)
        weights[name] = (score>=cut)&radial
    simple = mu[:,None]+width[:,None]*Z
    a,b,c = tables(t,p['labels'],selected,q,simple,weights)
    root = DEST/side
    root.mkdir(parents=True,exist_ok=True)
    csv_rows(root/'distance.csv',a)
    csv_rows(root/'migration.csv',b)
    csv_rows(root/'contaminants.csv',c)
    csv_rows(root/'relation.csv',[{key:rel[key][i] for key in rel} for i in range(len(rel['color']))])
    joblib.dump(dict(relation=rel,scale=scale,ablation=models),root/'models.joblib')
    atomic_npz(root/'fixed_rows.npz',**subset(dict(t,qrf_mu=q,simple_mu=simple,labels=p['labels'],
        **weights),selected))
    save_json(root/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        held_out=int(selected.sum()),color_training=int(train.sum()),color_training_gaia=int((train&good_ref).sum()),
        m30_rows_excluded=int(field.sum()),soft_scale=scale,outputs={f.name:sha(f) for f in root.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('side',choices=tuple(ROUTES))
    run(p.parse_args().side)

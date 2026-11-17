#!/usr/bin/env python3
"""Test optional native bands against grz on the same fit and evaluation stars."""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.data import SIGNIFICANT
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_catalog import OUT as EXPANSION
from v3_10_selection import references
from v3_11_layers import read_receipt
from v3_11_selection import DEST as LABELS
from v3_12_reference import redrock_and_physics, separation
from v3_14_population import BRANCH, DEST as V314, KNOWN_REGIONS, MIN_MASS, N, QUANTILES, mixture_quantiles

DEST = ROOT/'data/v3_15/bands'
DISTANCES = ((0.,2.),(2.,5.),(5.,10.),(10.,np.inf))
SCALES = {'grz':(.10,.06),'grzi':(.10,.06,.08),'grzW1W2':(.10,.06,.30,.14)}


def photometry(side,t,inputs):
    root = EXPANSION/side/'reference'
    flags = np.zeros(len(t['ls_id']),bool)
    states = np.full((len(flags),6),-1,dtype='i1')
    flux,error,trans = [np.full((len(flags),6),np.nan) for _ in range(3)]
    for mark in sorted(root.glob('[0-9][0-9].json')):
        r = json.loads(mark.read_text())
        for name,digest in r['outputs'].items():
            x = read_columns(verified(mark.parent/name,digest),
                    ('TYPE','ls_id','flux','flux_error','transmission','band_state'))
            j = unique_join(x['ls_id'],t['ls_id'])
            k = j>=0
            ix = j[k]
            if flags[ix].any() or len(np.unique(ix))!=len(ix) or not np.all(x['TYPE'][k]=='PSF'):
                raise ValueError('Nonunique or non-PSF native photometry')
            states[ix] = x['band_state'][k]
            flux[ix],error[ix],trans[ix] = [x[key][k] for key in ('flux','flux_error','transmission')]
            flags[ix] = True
        inputs[str(mark.relative_to(ROOT))] = sha(mark)
    if not flags.all():
        raise ValueError('Missing native flux for a fixed PSF identity')
    with np.errstate(invalid='ignore',divide='ignore'):
        mag = 22.5-2.5*np.log10(flux/trans)
        snr = flux/error
    gr = np.column_stack((mag[:,0]-mag[:,1],mag[:,1]-mag[:,3]))
    views = dict(grz=gr,grzi=np.column_stack((gr,mag[:,1]-mag[:,2])),
          grzW1W2=np.column_stack((gr,mag[:,1]-mag[:,4],mag[:,4]-mag[:,5])))
    opt = np.isfinite(gr).all(axis=1)&(states[:,[0,1,3]]==SIGNIFICANT).all(axis=1)
    eligible = dict(grz=opt,grzi=opt&(states[:,2]==SIGNIFICANT)&(snr[:,2]>=2.5/np.log(10)/.1),
                    grzW1W2=opt&(states[:,4:]==SIGNIFICANT).all(axis=1)&(snr[:,4:]>=5).all(axis=1))
    for key in views:
        eligible[key] &= np.isfinite(views[key]).all(axis=1)
    return views,eligible,states


class Neighborhood:
    def __init__(self,x,absolute,label):
        self.label = label
        self.absolute = absolute
        self.tree = cKDTree(x)
        self.global_n = min(2*N,len(label))
        self.idx = [np.flatnonzero(label==k) for k in range(len(BRANCH))]
        self.n = min([N]+[len(ids) for ids in self.idx if len(ids)])
        self.branches = [cKDTree(x[ids]) if len(ids) else None for ids in self.idx]

    def predict(self,x,r):
        _,i = self.tree.query(x,k=np.arange(1,self.global_n+1))
        p = np.column_stack([(self.label[i]==k).mean(axis=1) for k in range(len(BRANCH))])
        support = np.zeros_like(p,dtype=bool)
        repr_mr = np.zeros((len(x),len(BRANCH),self.n))
        for k,tree in enumerate(self.branches):
            if tree is None:
                continue
            d,j = tree.query(x,k=np.arange(1,self.n+1))
            support[:,k] = d[:,-1]<=2.
            repr_mr[:,k] = self.absolute[self.idx[k][j]]
        mass = (p*support).sum(axis=1)
        branch = np.divide(p*support,mass[:,None],out=np.zeros_like(p),where=mass[:,None]>0)
        q = r[:,None]-mixture_quantiles(repr_mr,branch)[:,::-1]
        q[mass<MIN_MASS] = np.nan
        conditional = r[:,None,None]-np.quantile(repr_mr,1-QUANTILES,axis=2).transpose(1,2,0)
        conditional[~support] = np.nan
        return q,conditional,p,mass


def forecast(view,x,train,ids,t,ref,label):
    scale = np.array(SCALES[view])
    model = Neighborhood(x[train]/scale,t['r0'][train]-10-5*np.log10(ref[train]),label[train])
    q,c,p,m = [],[],[],[]
    for k in range(0,len(ids),4096):
        j = ids[k:k+4096]
        pred,branches,probs,mass = model.predict(x[j]/scale,t['r0'][j])
        q.append(pred)
        p.append(probs)
        m.append(mass)
        # The explicit oracle is produced after the photometry-only prediction.
        known = label[j]
        y = branches[np.arange(len(j)),np.maximum(known,0)]
        y[known<0] = np.nan
        c.append(y)
    return dict(mu=np.concatenate(q),oracle=np.concatenate(c),probability=np.concatenate(p),
                usable_mass=np.concatenate(m),per_branch_neighbors=model.n)


def classes(label,probability,frame):
    rows,cal = [],[]
    hard = probability.argmax(axis=1)
    for k,name in enumerate(BRANCH):
        if not frame.any():
            continue
        true = label==k
        p = probability[:,k]
        n = frame&true
        rows.append(dict(branch=name,n=int(frame.sum()),true_n=int(n.sum()),
             brier=float(np.mean((p[frame]-true[frame])**2)),
             recall=float((hard[n]==k).mean()) if n.any() else None,
             mean_true_probability=float(p[n].mean()) if n.any() else None,
             predicted_n=int((frame&(hard==k)).sum()),
             false_positive=int((frame&~true&(hard==k)).sum())))
        for lo,hi in zip(np.linspace(0,1,11)[:-1],np.r_[np.linspace(.1,.9,9),1.00001]):
            s = frame&(p>=lo)&(p<hi)
            if s.any():
                cal.append(dict(branch=name,prob_lo=lo,prob_hi=min(1.,hi),
                                n=int(s.sum()),frequency=float(true[s].mean()),mean_p=float(p[s].mean())))
    return rows,cal


def paired_delta(baseline,alternate):
    delta = np.asarray(alternate,dtype=float)-baseline
    n = len(delta)
    draws = np.random.default_rng(31510).multinomial(n,np.full(n,1/n),size=512)
    lo,hi = np.quantile(draws@delta/n,(.025,.975))
    return dict(n=n,delta=float(delta.mean()),p025=float(lo),p975=float(hi))


def tables(t,label,views,qrf):
    rows,category,cal,deltas = [],[],[],[]
    base,alt = views.values()
    common = np.logical_and.reduce([np.isfinite(v['mu']).all(axis=1) for v in views.values()])
    qrf_ok = np.isfinite(qrf).all(axis=1)
    groups = [('all',np.ones(len(label),bool))]+[(k,label==i) for i,k in enumerate(BRANCH)]
    for source,d,flag in (
        ('Gaia',t['reference_kpc'],t['gaia_state']==6),('SpecDis',t['specdis_kpc'],t['specdis_quality']),
        ('SPDist',t['spdist_kpc'],t['spdist_quality'])):
        for lo,hi in DISTANCES:
            use = flag&(d>0)&(d>=lo)&(d<hi)
            for color,k in [('all',np.ones(len(label),bool)),('blue',(t['g0']-t['r0'])<.5)]:
                sel = use&k
                info = dict(reference=source,d_lo=lo,d_hi=hi,color=color)
                for method,x in views.items():
                    a,b = classes(label,x['probability'],sel&(label>=0))
                    category.extend(dict(**info,method=method,**r) for r in a)
                    cal.extend(dict(**info,method=method,**r) for r in b)
                forecasts = {n:x['mu'] for n,x in views.items()}
                forecasts.update({n+' true branch':x['oracle'] for n,x in views.items()})
                forecasts['frozen QRF'] = qrf
                for name,g in groups:
                    s = sel&g
                    if not s.any():
                        continue
                    for method,mu in forecasts.items():
                        have = np.isfinite(mu).all(axis=1)
                        k = s&common&qrf_ok&have
                        rows.append(dict(**info,branch=name,method=method,available=int(s.sum()),
                            common=int((s&common&qrf_ok).sum()),supported=int((s&have).sum()),
                            **distance_summary(mu[k],10+5*np.log10(d[k]))))
                if lo<2 or hi>10:
                    continue
                for i,(name,g) in enumerate(groups[1:]):
                    s = sel&(label>=0)
                    if s.any():
                        y = label[s]==i
                        deltas.append(dict(**info,branch=name,metric='Brier',**paired_delta(
                          (base['probability'][s,i]-y)**2,(alt['probability'][s,i]-y)**2)))
                    k = sel&g&common&qrf_ok
                    if k.any():
                        ref_mu = 10+5*np.log10(d[k])
                        deltas.append(dict(**info,branch=name,metric='severe fraction',**paired_delta(
                          abs(base['mu'][k,2]-ref_mu)>1,abs(alt['mu'][k,2]-ref_mu)>1)))
    return rows,category,cal,deltas


def run(side):
    t,inputs = references(side)
    roles = read_receipt(LABELS/side,'complete.json','scores.npz',
                        ('TYPE','ls_id','labels','new_train','held_out'),inputs)
    np.testing.assert_array_equal(roles['ls_id'],t['ls_id'])
    np.testing.assert_array_equal(roles['TYPE'],t['TYPE'])
    original = read_receipt(V314/side,'complete.json','heldout.npz',('TYPE','ls_id','branch'),inputs)
    old = unique_join(original['ls_id'],t['ls_id'])
    if np.any(old<0) or not np.all(original['TYPE']=='PSF') or not np.all(roles['held_out'][old]):
        raise ValueError('Changed v3.14 holdout role or morphology')
    redrock_and_physics(side,t,inputs)
    views,quality,states = photometry(side,t,inputs)
    restricted = quality['grz']&np.isfinite(t['r0'])
    for p in KNOWN_REGIONS:
        restricted &= separation(t['RA'],t['DEC'],p)>=2.
    gaia = (t['gaia_state']==6)&(t['reference_kpc']>0)
    ref = np.where(gaia,t['reference_kpc'],np.where(t['specdis_quality'],t['specdis_kpc'],np.nan))
    k = roles['labels']
    h = t['rvs_feh']
    label = np.select([(k==0)&np.isfinite(h)&(h>=-1.5),(k==0)&np.isfinite(h)&(h<-1.5),
                        k==2,k==1],np.arange(len(BRANCH)),default=-1)
    np.testing.assert_array_equal(label[old],original['branch'])
    parent = roles['new_train']&restricted&(label>=0)&np.isfinite(ref)&(ref>0)
    eval_role = np.zeros(len(parent),bool)
    eval_role[old] = True
    if (eval_role&parent).any():
        raise ValueError('Role leakage')
    out = DEST/side
    out.mkdir(parents=True,exist_ok=True)
    roster = []
    for alternate in ('grzi','grzW1W2'):
        train = parent&quality[alternate]
        test = eval_role&quality[alternate]&restricted
        ids = np.flatnonzero(test)
        for branch,name in zip(range(len(BRANCH)),BRANCH):
            roster.append(dict(pair=alternate,branch=name,train_parent=int((parent&(label==branch)).sum()),
              test_parent=int((eval_role&(label==branch)).sum()),train_pair=int((train&(label==branch)).sum()),
              test_pair=int((test&(label==branch)).sum())))
        if not len(ids) or not train.any():
            continue
        match = {view:forecast(view,views[view],train,ids,t,ref,label) for view in ('grz',alternate)}
        a = {key:val[ids] for key,val in t.items()}
        with np.errstate(divide='ignore',invalid='ignore'):
            qrf = 10+5*np.log10(a['qrf_quantiles_kpc'])
        distance,branch,cal,deltas = tables(a,label[ids],match,qrf)
        for name,records in [('distance',distance),('branch',branch),('calibration',cal),('paired_delta',deltas)]:
            csv_rows(out/f'{alternate}_{name}.csv',records)
        atomic_npz(out/f'{alternate}_paired.npz',TYPE=a['TYPE'],ls_id=a['ls_id'],true_branch=label[ids],
            train_TYPE=t['TYPE'][train],train_ls_id=t['ls_id'][train],bands=states[ids],
            **{f'{v}_{k}':x for v,d in match.items() for k,x in d.items()})
    csv_rows(out/'roster.csv',roster)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        heldout=len(old),grz_quality=int((eval_role&quality['grz']).sum()),
        i_and_wise=int((eval_role&quality['grzi']&quality['grzW1W2']).sum()),
        outputs={f.name:sha(f) for f in out.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('side',choices=('north','south'))
    run(parser.parse_args().side)

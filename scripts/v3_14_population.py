#!/usr/bin/env python3
"""Test empirical luminosity branches on fixed, independently held-out field stars."""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_10_selection import features, references
from v3_11_layers import read_receipt
from v3_11_selection import DEST as LABELS
from v3_12_reference import DEST as V312, Z, predict, redrock_and_physics, separation
from v3_12_structure import load_model

DEST = ROOT/'data/v3_14/population'
BRANCH = ('ordinary_dwarf','metal_poor_dwarf','intermediate','giant')
SCALE = np.array([.10,.06])
N = 64
MIN_MASS = .90
QUANTILES = np.array([.025,.16,.5,.84,.975])
KNOWN_REGIONS = ((259.281,43.136),(250.422,36.460),(325.092,-23.180),
                 (229.638,2.081),(189.8666,-26.7441))


def mixture_quantiles(absolute,probability):
    """Use a finite weighted mixture, retaining branch width and multimodality."""
    n,k,c = absolute.shape
    values = absolute.reshape(n,k*c)
    weights = np.broadcast_to(probability[:,:,None]/c,absolute.shape).reshape(n,k*c)
    index = np.argsort(values,axis=1)
    ranked = np.take_along_axis(values,index,axis=1)
    cum = np.cumsum(np.take_along_axis(weights,index,axis=1),axis=1)
    out = np.column_stack([ranked[np.arange(n),(cum>=q).argmax(axis=1)] for q in QUANTILES])
    out[cum[:,-1]<.999999] = np.nan
    return out


def predict_branches(x,all_x,absolute,labels,r):
    base = cKDTree(all_x)
    _,ids = base.query(x,k=2*N)
    p = np.column_stack([(labels[ids]==k).mean(axis=1) for k in range(len(BRANCH))])
    templates = np.empty((len(x),len(BRANCH),N))
    available = np.empty_like(p,dtype=bool)
    for k in range(len(BRANCH)):
        use = labels==k
        distance,j = cKDTree(all_x[use]).query(x,k=N)
        available[:,k] = distance[:,-1]<=2.
        templates[:,k] = absolute[use][j]
    kept = (p*available).sum(axis=1)
    weights = np.divide(p*available,kept[:,None],out=np.zeros_like(p),where=kept[:,None]>0)
    q = r[:,None]-mixture_quantiles(templates,weights)[:,::-1]
    q[kept<MIN_MASS] = np.nan
    conditional = r[:,None,None]-np.quantile(templates,1-QUANTILES,axis=2).transpose(1,2,0)
    conditional[~available] = np.nan
    return q,conditional,weights,kept


def audit_rows(t,branch,methods):
    rows = []
    color = t['g0']-t['r0']
    classes = [('all',np.ones(len(color),bool))]+[(name,branch==k) for k,name in enumerate(BRANCH)]
    common = np.logical_and.reduce([np.isfinite(v).all(axis=1) for n,v in methods.items()
                                   if n!='true branch diagnostic'])
    for source,d,flag in (
       ('Gaia',t['reference_kpc'],t['gaia_state']==6),
       ('SpecDis',t['specdis_kpc'],t['specdis_quality']),
       ('SPDist',t['spdist_kpc'],t['spdist_quality'])):
        for name,c in classes:
            for lo,hi in ((0.,2.),(2.,5.),(5.,10.),(10.,np.inf)):
                for a,b in ((-np.inf,np.inf),(-np.inf,.5),(.5,.8),(.8,1.4)):
                    base = c&flag&(d>0)&(d>=lo)&(d<hi)&(color>=a)&(color<b)
                    if not base.any():
                        continue
                    for model,q in methods.items():
                        tested = base&common&np.isfinite(q).all(axis=1)
                        rows.append(dict(reference=source,branch=name,d_lo=lo,d_hi=hi,gr_lo=a,gr_hi=b,
                            method=model,available=int(base.sum()),common=int((base&common).sum()),
                            supported=int((base&np.isfinite(q).all(axis=1)).sum()),
                            **distance_summary(q[tested],10+5*np.log10(d[tested]))))
    return rows


def run(side):
    t,inputs = references(side)
    old = read_receipt(LABELS/side,'complete.json','scores.npz',
                       ('TYPE','ls_id','labels','new_train','held_out'),inputs)
    np.testing.assert_array_equal(t['ls_id'],old['ls_id'])
    np.testing.assert_array_equal(t['TYPE'],old['TYPE'])
    if not np.all(t['TYPE']=='PSF'):
        raise ValueError('Unverified native morphology')
    redrock_and_physics(side,t,inputs)
    x = features(side,t,inputs)
    colors = x[:,:2] if side=='north' else x[:,-2:]
    finite = np.isfinite(colors).all(axis=1)&np.isfinite(t['r0'])
    mask = finite.copy()
    for center in KNOWN_REGIONS:
        mask &= separation(t['RA'],t['DEC'],center)>=2.
    spec = t['specdis_kpc']
    gaia = (t['gaia_state']==6)&(t['reference_kpc']>0)
    ref = np.where(gaia,t['reference_kpc'],np.where(t['specdis_quality'],spec,np.nan))
    known = old['labels']
    feh = t['rvs_feh']
    classification = np.select([
        (known==0)&np.isfinite(feh)&(feh>=-1.5),(known==0)&np.isfinite(feh)&(feh<-1.5),
        known==2,known==1],np.arange(len(BRANCH)),default=-1)
    tr = old['new_train']&mask&(classification>=0)&np.isfinite(ref)&(ref>0)
    te = old['held_out']&mask
    ix = np.flatnonzero(te)
    if np.any(np.bincount(classification[tr],minlength=len(BRANCH))<N):
        raise ValueError('Insufficient physical training branches')
    q = np.empty((len(ix),5))
    cond = np.empty_like(q)
    support = np.empty(len(ix))
    prob = np.empty((len(ix),len(BRANCH)))
    for start in range(0,len(ix),4096):
        k = slice(start,start+4096)
        j = ix[k]
        lum = t['r0'][tr]-10-5*np.log10(ref[tr])
        q[k],cs,prob[k],support[k] = predict_branches(colors[j]/SCALE,colors[tr]/SCALE,
                                                     lum,classification[tr],t['r0'][j])
        ids = classification[j]
        cond[k] = cs[np.arange(len(j)),np.maximum(ids,0)]
        cond[np.arange(start,start+len(j))[ids<0]] = np.nan
    model = load_model(V312/side,'models.joblib',inputs)['relation']
    mu,width = predict(model,t['g0'][ix]-t['r0'][ix],t['r0'][ix])
    a = subset(t,ix)
    with np.errstate(divide='ignore',invalid='ignore'):
        methods = {'frozen QRF':10+5*np.log10(a['qrf_quantiles_kpc']),
                'one-color relation':mu[:,None]+width[:,None]*Z,
                'grz marginalized branches':q,'true branch diagnostic':cond}
    rows = audit_rows(a,classification[ix],methods)
    out = DEST/side
    out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'distance.csv',rows)
    atomic_npz(out/'heldout.npz',TYPE=a['TYPE'],ls_id=a['ls_id'],branch=classification[ix],
        usable_training_mass=support,branch_weights=prob,methods=np.array(list(methods)),
        mu=np.stack(list(methods.values())))
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        held_out=len(ix),training=dict(zip(BRANCH,np.bincount(classification[tr],minlength=len(BRANCH)))),
        outputs={f.name:sha(f) for f in out.iterdir() if f.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('side',choices=('north','south'))
    run(parser.parse_args().side)

"""Frozen inputs and numerical helpers for the 2026-11-18 science experiments."""
import csv
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.stats import norm

from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from paper_v2_reference import metric
from paper_v2_scan import OUT as PRIOR, CLASSES
from v3_8_direct_figures import BLUE, ORANGE, GREEN, plt, save
from paper_v1_figures import PAPER
from v3_10_figures import check_templates

CONFIG=ROOT/'configs/science_validation_20261118.json'
P=json.loads(CONFIG.read_text())
DEST=ROOT/'data/science_validation_20261118'
RNG=np.random.default_rng(P['seed'])


def read(path, fields, inputs):
    marker=path/'complete.json'
    r=json.loads(marker.read_text())
    inputs[str(marker.relative_to(ROOT))]=sha(marker)
    return read_columns(verified(path/fields[0],r['outputs'][fields[0]]),fields[1:])


def cached(inputs, fields=None):
    marker=PRIOR/'reference/complete.json'
    r=json.loads(marker.read_text())
    inputs['reference_census']=sha(marker)
    path=verified(marker.with_name('reference_rows.npz'),r['outputs']['reference_rows.npz'])
    if fields is None:
        with np.load(path) as z:
            t={k:z[k] for k in z.files}
    else:
        t=read_columns(path,fields)
    assert np.all(t['TYPE']=='PSF')
    return t


def out(stage):
    d=DEST/stage
    d.mkdir(exist_ok=True,parents=True)
    return d


def complete(d, inputs, **metadata):
    save_json(d/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        config=sha(CONFIG),style=check_templates(),**metadata,
        outputs={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='complete.json'}))


def sky(t):
    return hp.ang2pix(P['sky_nside'],t['RA'],t['DEC'],lonlat=True,nest=True)


def interval(values):
    a=np.asarray(values)
    a=a[np.isfinite(a)]
    if not len(a):
        return dict(ci_lo=None,ci_hi=None,valid_draws=0)
    x=np.quantile(a,[.025,.975])
    return dict(ci_lo=x[0],ci_hi=x[1],valid_draws=len(a))


def index_join(ids, keys):
    order=np.argsort(keys,kind='stable')
    k=np.searchsorted(keys[order],ids)
    good=k<len(keys)
    good[good] &= keys[order[k[good]]]==ids[good]
    result=np.full(len(ids),-1,'i8')
    result[good]=order[k[good]]
    return result


def quantile_distance(mass, edges):
    c=np.cumsum(mass,axis=1)
    c=np.divide(c,c[:,-1:],out=np.zeros_like(c),where=c[:,-1:]>0)
    q=[]
    for u in (.025,.16,.5,.84,.975):
        j=(c>=u).argmax(axis=1)
        lo=np.where(j>0,c[np.arange(len(c)),np.maximum(j-1,0)],0)
        hi=c[np.arange(len(c)),j]
        f=np.divide(u-lo,hi-lo,out=np.zeros_like(lo),where=hi>lo)
        m=edges[np.arange(len(c)),j]+f*(edges[np.arange(len(c)),j+1]-edges[np.arange(len(c)),j])
        q.append(10**((m-10)/5))
    return np.array(q).T


def block_metrics(q,d,block):
    result=metric(q,d)
    keys=np.unique(block)
    stats=[]
    if len(keys)>1:
        for _ in range(P['bootstrap_draws']):
            counts=np.bincount(RNG.integers(len(keys),size=len(keys)),minlength=len(keys))
            ix=np.concatenate([np.flatnonzero(block==k).repeat(n) for k,n in zip(keys,counts) if n])
            stats.append(metric(q[ix],d[ix]))
    result['sky_blocks']=len(keys)
    for key in ('bias','nmad','severe','c68'):
        z=interval([x[key] for x in stats])
        result.update({key+'_lo':z['ci_lo'],key+'_hi':z['ci_hi']})
    return result


def figure(d,name,fig,rows,arrays=None):
    with plt.rc_context(PAPER):
        save(fig,d,name,rows,arrays)

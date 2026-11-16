#!/usr/bin/env python3
"""Validate the v3.13 fields, compare soft weights and plot member CMD residuals."""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.controls import probe_features
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import STYLE, plt, save
from v3_9_references import predictor_features
from v3_10_figures import check_templates
from v3_11_selection import DEST as REVISED
from v3_12_structure import class_probability, load_model
from v3_13_clusters import DEST as CLUSTERS, SYSTEMS, DISTANCE, GR, MAG, MR, photos
from v3_13_scale import DEST as SCALE

DEST = ROOT/'data/v3_13/delivery'


def bind(path,inputs):
    meta = json.loads(path.read_text())
    for key,digest in meta['outputs'].items():
        verified(path.parent/key,digest)
    inputs[str(path.relative_to(ROOT))] = sha(path)
    return meta


def cmd(name,t,path):
    with np.errstate(invalid='ignore',divide='ignore'):
        q = 10+5*np.log10(t['qrf_kpc'][:,2])
    mu = 10+5*np.log10(SYSTEMS[name][1])
    gr,mr = t['g0']-t['r0'],t['Mr_reference']
    rows, data = [], {}
    with plt.rc_context(STYLE):
        f,ax = plt.subplots(2,2,figsize=(13,10),layout='constrained',sharex=True,sharey=True)
        for i,(method,x) in enumerate((('QRF',q),('Color relation',t['linear_mu']))):
            k = t['target']&t['member']&np.isfinite(x)
            num = np.histogram2d(gr[k],mr[k],bins=(GR,MR))[0]
            bias = np.histogram2d(gr[k],mr[k],bins=(GR,MR),weights=mu-x[k])[0]
            val = np.divide(bias,num,out=np.zeros_like(num),where=num>0)
            img = ax[i,0].pcolormesh(GR,MR,np.ma.masked_less(num.T,1),cmap='viridis',rasterized=True)
            f.colorbar(img,ax=ax[i,0],pad=.02,label='Members per cell')
            img = ax[i,1].pcolormesh(GR,MR,np.ma.masked_where(num.T<3,val.T),cmap='RdBu_r',
                                     vmin=-.75,vmax=.75,rasterized=True)
            f.colorbar(img,ax=ax[i,1],pad=.02,label='Mean predicted minus reference absolute r (mag)')
            for a,panel in zip(ax[i],('Member count','Luminosity residual')):
                a.set(xlabel='Dereddened g − r (mag)',ylabel='Absolute r at cluster distance',
                      title=f'{method} · {panel}',xlim=(-.3,1.4),ylim=(9,-2))
            data[f'count_{i}'],data[f'delta_{i}'] = num,val
            rows.append(dict(method=method,matched_forecasts=int(k.sum()),visible_cell_sum=int(num.sum())))
        f.suptitle(f'{name} · luminosity residuals along the matched member CMD')
        save(f,path,'cmd_residuals',rows,dict(gr=GR,Mr=MR,**data))


def weights(name,t,path,inputs):
    side = 'south' if name=='M30' else 'north'
    fit = load_model(REVISED/side,'model.joblib',inputs)
    more = ('flux','flux_error','transmission','band_mask') if side=='south' else ()
    p,_ = photos(name,inputs,more)
    j = unique_join(t['ls_id'],p['ls_id'])
    if np.any(j<0) or not np.all(p['TYPE'][j]=='PSF'):
        raise ValueError('The weight comparison changed native admission')
    p = {k:v[j] for k,v in p.items()}
    features = probe_features(p,'grz') if side=='south' else predictor_features(p['mag_grz'])
    s = class_probability(fit,features)
    s[t['training_or_calibration']] = 0.
    area = np.array([float(row['target_deg2' if q==0 else 'control_deg2'])
         for q in (0,1) for row in first_row(CLUSTERS/name/'area.csv')])
    rows = []
    dist = SYSTEMS[name][1]
    w = np.clip((np.minimum(DISTANCE[1:],dist+1)-np.maximum(DISTANCE[:-1],dist-1))/.25,0,1)
    middle = (DISTANCE[:-1]+DISTANCE[1:])/2
    pdf = t['qrf_mass'][:,1:-1]
    for lo,hi in MAG:
        tier = (t['r']>=lo)&(t['r']<hi)&np.isfinite(t['qrf_kpc'][:,2])
        on,off = tier&t['target'],tier&~t['target']
        for key,score in (('support',np.ones(len(s))),('soft dwarf',s)):
            x,y = [np.einsum('ij,i->j',pdf[k],score[k])/a for k,a in zip((on,off),area)]
            net = x-y
            m = on&t['member']
            rows.append(dict(r_min=lo,r_max=hi,weight=key,target_rows=int(on.sum()),
                control_rows=int(off.sum()),confirmed_members=int(m.sum()),member_weight=score[m].sum(),
                control_weight=score[off].sum(),target_density_window=x@w,control_density_window=y@w,
                excess_window=net@w,peak_kpc=middle[np.argmax(net)] if np.max(net)>0 else None))
    csv_rows(path/'soft_depth.csv',rows)
    atomic_npz(path/'soft_weights.npz',TYPE=t['TYPE'],ls_id=t['ls_id'],dwarf_score=s)


def first_row(path):
    with path.open() as f:
        return [next(csv.DictReader(f))]


def figure_check(path):
    with Image.open(path) as im:
        if min(im.size)<1000 or abs(im.info['dpi'][0]-300)>1:
            raise ValueError(f'Figure format changed: {path}')
        im.verify()
    with path.with_suffix('.pdf').open('rb') as f:
        if f.read(5)!=b'%PDF-':
            raise ValueError('Missing PDF plate')


def run(name):
    inputs = {}
    source = CLUSTERS/name
    receipt = bind(source/'complete.json',inputs)
    if receipt['inputs']['style']!=check_templates():
        raise ValueError('Changed style')
    t = read_columns(source/'sources.npz',('TYPE','ls_id','target','member','g0','r0','r',
        'Mr_reference','linear_mu','qrf_kpc','qrf_mass','training_or_calibration'))
    if not np.all(t['TYPE']=='PSF') or len(np.unique(t['ls_id']))!=receipt['admitted']:
        raise ValueError('Invalid source handoff')
    has = np.isfinite(t['qrf_kpc'][:,2])
    if (has&t['training_or_calibration']).any():
        raise ValueError('A frozen distance training identity has leaked')
    np.testing.assert_allclose(t['qrf_mass'].sum(axis=1),has.astype(float),rtol=0,atol=2e-7)
    path = DEST/name
    path.mkdir(parents=True,exist_ok=True)
    cmd(name,t,path)
    weights(name,t,path,inputs)
    for area in (path,source):
        for image in area.glob('*.png'):
            figure_check(image)
    save_json(path/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,n=int(has.sum()),
        outputs={p.name:sha(p) for p in path.iterdir() if p.name!='complete.json'}))


def final():
    inputs = {}
    for name in SYSTEMS:
        bind(DEST/name/'complete.json',inputs)
    for side in ('north','south'):
        meta = bind(SCALE/side/'complete.json',inputs)
        k = read_columns(SCALE/side/'identity.npz',('TYPE','ls_id','dual_fit','dual_cal','eval_role','dual_eval'))
        if not np.all(k['TYPE']=='PSF') or len(np.unique(k['ls_id']))!=len(k['ls_id']):
            raise ValueError('Duplicate or invalid teacher-swap identity')
        if (k['dual_fit']&k['dual_cal']).any() or ((k['dual_fit']|k['dual_cal'])&k['eval_role']).any():
            raise ValueError('Teacher-swap roles are not separated')
        assert int(k['dual_fit'].sum())==meta['fit']['Gaia paired']==meta['fit']['SpecDis paired']
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        style=check_templates(),figures=len(SYSTEMS)*3,outputs={}))


if __name__=='__main__':
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',choices=(*SYSTEMS,'final'))
    name=parser.parse_args().name
    if name=='final':
        final()
    else:
        run(name)

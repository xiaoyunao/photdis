#!/usr/bin/env python3
"""Blind three-method GD-1 inference with same-source spatial resampling."""
import csv
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter

from photdis.stage1 import require_compute
from v2_1_near import atomic_npz, csv_rows, save_json, sha
from v3_11_layers import read_receipt
from v3_13_clusters import SYSTEMS
from v3_14_audit import DEST as AUDIT
from v3_16_gd1 import CONFIG, DEST, EDGES, LIMITS, NSEG, NSTEP, OFFSETS, P
from v3_16_sample import APPARENT, COLOR, D

ABS = np.arange(-2.,9.01,.1)
MU = 10+5*np.log10(D)
METHODS = ('Frozen QRF','M13 CMD','M92 deep CMD')
DEPTH = {'r < 20':(0,), '20 <= r < 21':(1,), '21 <= r < 22':(2,),
         'r < 21':(0,1), 'r < 22':(0,1,2)}
DRAWS = P['spatial_draws']
FIT = DEST/'fit_supported'


def cluster_template(name,inputs):
    base = read_receipt(AUDIT/name,'complete.json','native_join.npz',
        ('TYPE','g0','r0','r','target','member','low_crowding','training_or_calibration'),inputs)
    assert (base['TYPE']=='PSF').all()
    color = base['g0']-base['r0']
    mag = base['r0']-10-5*np.log10(SYSTEMS[name][1])
    good = base['low_crowding']&(base['r']<22)&~base['training_or_calibration']
    on,off = good&base['target'],good&~base['target']
    def counts(k):
        return np.histogram2d(color[k],mag[k],(COLOR,ABS))[0]
    if name=='M13':
        active = on&base['member']
        h = counts(active)
        meta = {'source':'high-probability native members','target':int(active.sum())}
    else:
        area = AUDIT/name/'area.csv'
        expected = json.loads((area.parent/'complete.json').read_text())['outputs']['area.csv']
        if sha(area)!=expected:
            raise ValueError('Changed cluster template exposure')
        with area.open() as f:
            a = list(csv.DictReader(f))
        geo = [sum(float(row['area_deg2']) for row in a if
                   row['limiting_r']=='0' and row['zone']==zone) for zone in ('target','background')]
        h = counts(on)-(geo[0]/geo[1])*counts(off)
        meta = {'source':'native target minus background','target':int(on.sum()),
                'background':int(off.sum()),'native_target_area':geo[0],'native_background_area':geo[1]}
    width = P['template_width_mag']
    h = gaussian_filter(h,(width/np.diff(COLOR)[0],width/np.diff(ABS)[0]))
    h = np.clip(h,0,None)
    if h.sum()<=0:
        raise ValueError('Empty cluster template')
    h /= h.sum()
    meta['mass_Mr_ge_6'] = float(h[:,ABS[:-1]>=6].sum())
    return h,meta


def project(template):
    x = 0.5*(ABS[1:]+ABS[:-1])
    app = 0.5*(APPARENT[1:]+APPARENT[:-1])
    u = np.array([[np.interp(app-mu,x,h,left=0.,right=0.) for h in template] for mu in MU])
    return u.reshape(len(MU),-1)


def magnitude_response(extinction):
    """Fraction of each dereddened r bin in each observed-r tier."""
    edges = np.r_[-np.inf,LIMITS]
    width = np.diff(APPARENT)
    # The narrow attenuation range makes these quantiles sufficient for the
    # window response; retain the measured distribution rather than shifting r.
    shifts = np.quantile(extinction,(np.arange(101)+.5)/101)
    limits = (edges[:,None,None]-APPARENT[None,None,:-1]-shifts[None,:,None])/width
    cdf = np.clip(limits,0,1).mean(axis=1)
    response = np.diff(cdf,axis=0)
    return np.tile(response,(1,len(COLOR)-1))


def arrays(t):
    cell = (t['block'].astype('i8')*len(OFFSETS)+t['ribbon'])*len(LIMITS)+t['tier']
    shape = (NSEG,NSTEP,len(OFFSETS),len(LIMITS))
    n = int(np.prod(shape))
    f = t['qrf'][:,1:-1].astype('f8')
    q = np.zeros((n,len(D)))
    v = np.zeros_like(q)
    np.add.at(q,cell,f)
    np.add.at(v,cell,f*f)
    g = np.searchsorted(COLOR,t['color'],side='right')-1
    a = np.searchsorted(APPARENT,t['r0'],side='right')-1
    cols = len(COLOR)-1
    rows = len(APPARENT)-1
    h = np.bincount((cell*cols+g)*rows+a,minlength=n*cols*rows)
    return (h.reshape((*shape,cols*rows)),q.reshape((*shape,len(D))),v.reshape((*shape,len(D))),
            np.bincount(cell,minlength=n).reshape(shape))


def contrast(values,geo,j,keep,draw,variance=False):
    a = draw@geo
    on = draw@values[:,j]
    off = draw@values[:,keep].sum(axis=1)
    denominator = a[:,keep].sum(axis=1)
    alpha = np.divide(a[:,j],denominator,out=np.zeros_like(denominator),where=denominator>0)
    if variance:
        return on+alpha[:,None]**2*off
    return on-alpha[:,None]*off


def fit_curves(h,q,v,area,eligible,j,draw,templates,depth):
    # Retain magnitude-tier exposure terms when combining independent tiers.
    count,noise = np.zeros((len(draw),len(D))),np.zeros((len(draw),len(D)))
    num = np.zeros((templates.shape[1],len(draw),len(D)))
    den = np.zeros_like(num)
    for k in depth:
        bg = eligible[:,k].copy()
        bg[j] = False
        if j!=0:
            bg[0] = False
        geo = area[:,:,k]
        if not eligible[j,k] or bg.sum()<P['minimum_controls']-int(j!=0):
            return None
        count += contrast(q[:,:,k],geo,j,bg,draw)
        noise += contrast(v[:,:,k],geo,j,bg,draw,variance=True)
        signal = contrast(h[:,:,k],geo,j,bg,draw)
        var = np.maximum(contrast(h[:,:,k],geo,j,bg,draw,variance=True),5.)
        for i,u in enumerate(templates[k]):
            num[i] += (signal/var)@u.T
            den[i] += (1/var)@(u*u).T
    z = np.divide(count,np.sqrt(noise),out=np.full_like(count,np.nan),where=noise>0)
    cmd = np.divide(num,np.sqrt(den),out=np.full_like(num,np.nan),where=den>0)
    score = np.concatenate((z[None],cmd))
    peaks = np.concatenate((count[None],cmd))
    return peaks,score


def peak(curve,score):
    finite = np.isfinite(curve)&np.isfinite(score)
    index = np.argmax(np.where(finite,curve,-np.inf),axis=-1)
    valid = finite.any(axis=-1)
    z = np.take_along_axis(score,index[...,None],axis=-1)[...,0]
    return np.where(valid,D[index],np.nan),np.where(valid,z,np.nan)


def run():
    inputs = {'geometry':sha(CONFIG)}
    cov = read_receipt(DEST/'footprint','complete.json','area.npz',('area','coverage_ok','passed'),inputs)
    t = read_receipt(DEST/'sample','complete.json','sources.npz',
                    ('TYPE','ls_id','r','r0','color','qrf','block','ribbon','tier'),inputs)
    assert np.all(t['TYPE']=='PSF') and len(np.unique(t['ls_id']))==len(t['ls_id'])
    np.testing.assert_allclose(t['qrf'].sum(axis=1,dtype='f8'),1.,atol=2e-7,rtol=0)
    h,q,v,n = arrays(t)
    sources = [cluster_template(name,inputs) for name in ('M13','M92')]
    response = magnitude_response(t['r']-t['r0'])
    base_templates = np.array([project(x[0]) for x in sources])
    templates = base_templates[None]*response[:,None,None,:]
    area = cov['area'][1:].reshape(len(LIMITS),NSEG,NSTEP,len(OFFSETS)).transpose(1,2,3,0)
    eligible = cov['coverage_ok'].transpose(1,2,0)
    rng = np.random.default_rng(P['seed'])
    weights = np.concatenate((np.ones((NSEG,1,NSTEP)),
        rng.multinomial(NSTEP,np.full(NSTEP,1/NSTEP),size=(NSEG,DRAWS))),axis=1)
    rows,controls,content = [],[],{}
    for s in range(NSEG):
        for k,(name,depth) in enumerate(DEPTH.items()):
            if not cov['passed'][list(depth),s].all():
                continue
            curves,scores = fit_curves(h[s],q[s],v[s],area[s],eligible[s],0,weights[s],templates,depth)
            d,z = peak(curves,scores)
            null = []
            for j in range(1,len(OFFSETS)):
                trial = fit_curves(h[s],q[s],v[s],area[s],eligible[s],j,weights[s,:1],templates,depth)
                if trial is None:
                    continue
                x,y = peak(*trial)
                null.append(y[:,0])
                for m,method in enumerate(METHODS):
                    controls.append(dict(phi1=(EDGES[s]+EDGES[s+1])/2,depth=name,method=method,
                        phi2_offset=OFFSETS[j],peak=x[m,0],peak_z=y[m,0]))
            null = np.asarray(null)
            content[f's{s}_k{k}_peak'] = d
            content[f's{s}_k{k}_z'] = z
            content[f's{s}_k{k}_curves'] = curves[:,0]
            content[f's{s}_k{k}_scores'] = scores[:,0]
            for m,method in enumerate(METHODS):
                lo,hi = np.nanquantile(d[m,1:],(.16,.84))
                highest = np.nanmax(null[:,m])
                rows.append(dict(phi1=(EDGES[s]+EDGES[s+1])/2,depth=name,method=method,
                    target_sources=int(n[s,:,0,:][:,depth].sum()),
                    target_exposure_min_deg2=area[s,:,0,:].sum(axis=0)[list(depth)].min(),
                    peak=d[m,0],p16=lo,p84=hi,peak_z=z[m,0],control_n=len(null),
                    largest_control_z=highest,signal=bool(z[m,0]>=3 and z[m,0]>highest)))
    out = FIT
    out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'segments.csv',rows)
    csv_rows(out/'controls.csv',controls)
    atomic_npz(out/'curves.npz',distance=D,resample_blocks=weights,magnitude_response=response,template_m13=sources[0][0],
               template_m92=sources[1][0],color_edges=COLOR,absolute_edges=ABS,**content)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        templates={name:x[1] for name,x in zip(('M13','M92'),sources)},
        outputs={k:sha(out/k) for k in ('segments.csv','controls.csv','curves.npz')}))


if __name__=='__main__':
    require_compute()
    run()

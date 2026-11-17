#!/usr/bin/env python3
"""Evaluate frozen distance methods only in prequalified GD-1 segments."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_references import subset
from v3_10_working_sample import shell_mass
from v3_11_layers import read_receipt
from v3_16_fit import ABS, arrays, cluster_template, magnitude_response, peak, project
from v3_16_gd1 import EDGES, NSEG, NSTEP, P
from v3_16_sample import COLOR, D, D_EDGES
from v3_17_hess import weights
from v3_17_members import table
from v3_17_sample import C, CONFIG, DEST, NEW, NORTH

METHODS = ('Frozen QRF','M13 CMD','M92 CMD','Dotter CMD')
FAR = [1,2,3,4]
BLANK = [5,6,7,8]
CUTS = np.r_[-np.inf,10+5*np.log10(D_EDGES),np.inf]


def replay(t,inputs):
    home = DEST/'sample'/'complete.json'
    j = json.loads(home.read_text())['inputs']
    fits = [ROOT/x for x in j if '/predict/' in x and x.endswith('complete.json')]
    ids = t['ls_id']
    def scan(path):
        meta = json.loads(verified(path,j[str(path.relative_to(ROOT))]).read_text())
        result = []
        for name,h in meta['outputs'].items():
            if name.startswith('state_') or 'state_'+name not in meta['outputs']:
                continue
            mark = 'state_'+name
            r = read_columns(verified(path.parent/mark,meta['outputs'][mark]),('ls_id','TYPE','state'))
            x = np.flatnonzero(r['state']==3)
            x = x[np.isin(r['ls_id'][x],ids)]
            if not len(x):
                continue
            assert np.all(r['TYPE'][x]=='PSF')
            p = read_columns(verified(path.parent/name,h),('ls_id','TYPE','qrf_mu_edges','qrf_mu_mass'))
            m = unique_join(r['ls_id'][x],p['ls_id'])
            assert (m>=0).all() and np.all(p['TYPE'][m]=='PSF')
            y = unique_join(p['ls_id'][m],ids)
            b = shell_mass(p['qrf_mu_edges'][m],p['qrf_mu_mass'][m],CUTS)
            np.testing.assert_allclose(b.sum(axis=1),1.,rtol=0,atol=2e-7)
            result.append((y,b))
        return result
    b = np.zeros((len(ids),len(D)+2))
    seen = np.zeros(len(ids),bool)
    with ThreadPoolExecutor(max_workers=min(6,int(os.getenv('SLURM_CPUS_PER_TASK',1)))) as pool:
        for parts in pool.map(scan,fits):
            for i,mass in parts:
                if seen[i].any() or len(np.unique(i))!=len(i):
                    raise ValueError('Repeated QRF prediction during replay')
                b[i],seen[i] = mass,True
    np.testing.assert_array_equal(seen,t['qrf'])
    inputs['frozen_qrf_parts']=len(fits)
    return b


def physical_template(luminosity):
    iso = np.sort(table('v4_Figure3_dotter_isochrone.fits'),order='Mr')
    m = .5*(ABS[1:]+ABS[:-1])
    rgb = .5*(COLOR[1:]+COLOR[:-1])
    g = np.interp(m,iso['Mr'],iso['g-r'])
    band = np.exp(-.5*((rgb[:,None]-g)/P['template_width_mag'])**2)
    band /= np.maximum(band.sum(axis=0),np.finfo(float).tiny)
    lo,hi = C['ridge_Mr_range']
    cmd = band*luminosity[None,:]*((m>=lo)&(m<=hi))
    return cmd/cmd.sum()


def fitted(h,q,v,exposure,projected,draw,j):
    zq,eq = [np.zeros((len(draw),len(D))) for _ in range(2)]
    num,den = [np.zeros((projected.shape[1],len(draw),len(D))) for _ in range(2)]
    permitted = np.ones(len(draw),bool)
    for k in range(exposure.shape[-1]):
        a = draw@exposure[:,:,k]
        out = [weights(x,NEW,FAR,NEW[j]) if x[j]>0 else None for x in a]
        good = np.array([x is not None for x in out])
        if not good[0]:
            return None
        permitted &= good
        w = np.stack([x if x is not None else np.zeros(len(NEW)) for x in out])*a[:,j,None]
        def contrast(data,var):
            result = np.einsum('ds,sfc->dfc',draw,data[:,:,k])
            fitted = np.einsum('df,dfc->dc',w*w if var else w,result)
            if var:
                return result[:,j]+fitted
            return result[:,j]-fitted
        eq += contrast(q,False)
        zq += contrast(v,True)
        signal = contrast(h,False)
        var = np.maximum(contrast(h,True),5.)
        for p,u in enumerate(projected[k]):
            num[p] += (signal/var)@u.T
            den[p] += (1/var)@(u*u).T
    zq = np.divide(eq,np.sqrt(zq),out=np.full_like(eq,np.nan),where=zq>0)
    zc = np.divide(num,np.sqrt(den),out=np.full_like(num,np.nan),where=den>0)
    signal = np.concatenate((eq[None],zc))
    z = np.concatenate((zq[None],zc))
    signal[:,~permitted],z[:,~permitted] = np.nan,np.nan
    return peak(signal,z)


def run():
    hess = DEST/'hess'/'complete.json'
    gate = json.loads(hess.read_text())
    if not gate['authorize_distance_stage']:
        raise ValueError('Unmet GD-1 detection gate')
    use_segments = sorted(gate['signal_segments'])
    inp = {'config':sha(CONFIG),'detection':sha(hess),'segments':use_segments}
    prepare = DEST/'sample'
    fld = ('TYPE','ls_id','phi1','block','ribbon','tier','r','r0','color','qrf')
    t = read_receipt(prepare,'complete.json','desi.npz',fld,inp)
    cov = read_receipt(prepare,'complete.json','area.npz',('desi','desi_good'),inp)
    assert np.all(t['TYPE']=='PSF')
    t = subset(t,np.isin(t['block']//NSTEP,use_segments))
    pdf = replay(t,inp)
    s = t['qrf']
    # The original Boolean support becomes the rebinned archived PDF.
    t['qrf'] = pdf
    area = cov['desi'].reshape(3,NSEG,NSTEP,len(NEW)).transpose(1,2,3,0)
    good = cov['desi_good'].transpose(1,2,0)
    area = np.where(good[:,None,:,:],area,0.)
    m13,_ = cluster_template('M13',inp)
    m92,_ = cluster_template('M92',inp)
    physical = physical_template(m92.sum(axis=0))
    templates = np.array([project(x) for x in (m13,m92,physical)])
    response = magnitude_response(t['r']-t['r0'])
    projected = templates[None]*response[:,None,None,:]
    rng = np.random.default_rng(P['seed']+5)
    draw = np.concatenate((np.ones((NSEG,1,NSTEP)),
        rng.multinomial(NSTEP,np.full(NSTEP,1/NSTEP),size=(NSEG,P['spatial_draws']))),axis=1)
    rows,details = [],{}
    for cohort,sample in [('same_qrf_source',subset(t,s)),('all_native_cmd',t)]:
        h,q,v,n = arrays(sample)
        for seg in use_segments:
            a = area[seg]
            d,z = fitted(h[seg],q[seg],v[seg],a,projected,draw[seg],0)
            blank = [fitted(h[seg],q[seg],v[seg],a,projected,np.ones((1,NSTEP)),j) for j in BLANK]
            blank = np.asarray([b[1][:,0] for b in blank if b is not None])
            last = np.max(abs(blank),axis=0)
            for i,name in enumerate(METHODS):
                if i==0 and cohort=='all_native_cmd':
                    continue
                lo,hi = np.nanquantile(d[i,1:],(.16,.84))
                rows.append(dict(cohort=cohort,segment=seg,phi1=.5*(EDGES[seg]+EDGES[seg+1]),
                      split='exploration' if seg in C['exploration_segments'] else 'validation',
                      method=name,psf=int(n[seg,:,0].sum()),distance_kpc=d[i,0],p16_kpc=lo,p84_kpc=hi,
                      z=z[i,0],blank_abs_max=last[i],distance_signal=bool(z[i,0]>=C['minimum_z'] and z[i,0]>last[i])))
            details[f'{cohort}_{seg}_distance']=d
            details[f'{cohort}_{seg}_z']=z
    out = DEST/'distance'
    out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'fits.csv',rows)
    reference = verified(DEST/'hess'/'references.csv',gate['outputs']['references.csv'])
    inp['reference_csv'] = sha(reference)
    with reference.open() as f:
        ref = {int(x['segment']):x for x in csv.DictReader(f)}
    for r in rows:
        d = r['distance_kpc']
        for name,col in [('cfht','cfht_kpc'),('sdss','sdss_interpolated_kpc'),('desi','desi_curve_kpc')]:
            scale = float(ref[r['segment']][col])
            delta = d/scale-1
            r[name+'_relative_bias'] = delta
            r[name+'_ten_percent_with_signal'] = bool(r['distance_signal'] and abs(delta)<=P['reference_tolerance_fraction'])
    csv_rows(out/'scores.csv',rows)
    gradient = []
    for cohort,method in sorted({(x['cohort'],x['method']) for x in rows}):
        selected = [x for x in rows if (x['cohort'],x['method'],x['split'])==(cohort,method,'validation')]
        if len(selected)<2:
            continue
        a,b = sorted(selected,key=lambda x:x['phi1'])[::len(selected)-1]
        interval = b['phi1']-a['phi1']
        i = METHODS.index(method)
        draw_slopes = (details[f"{cohort}_{b['segment']}_distance"][i,1:]-
                       details[f"{cohort}_{a['segment']}_distance"][i,1:])/interval
        lo,hi = np.nanquantile(draw_slopes,(.025,.975))
        row = dict(cohort=cohort,method=method,validation_segments=len(selected),
             validated_distance_peaks=sum(x['distance_signal'] for x in selected),
             endpoint_secant=(b['distance_kpc']-a['distance_kpc'])/interval,p025=lo,p975=hi,
             multiple_segment_gradient_ready=sum(x['desi_ten_percent_with_signal'] for x in selected)>=3)
        for name,col in [('cfht','cfht_kpc'),('sdss','sdss_interpolated_kpc'),('desi','desi_curve_kpc')]:
            row[name+'_secant'] = (float(ref[b['segment']][col])-float(ref[a['segment']][col]))/interval
        gradient.append(row)
    csv_rows(out/'gradient.csv',gradient)
    atomic_npz(out/'distance.npz',draw=draw,physical_template=physical,**details)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inp,
        max_validation_segments=sum(i in C['validation_segments'] for i in use_segments),
        outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()

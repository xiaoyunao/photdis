#!/usr/bin/env python3
"""Retain full-tail CDF, marginal density and CRPS integration caches."""
from seq7_common import *
from seq7_math import evaluate_mix,uniform_curve,curve_crps
from seq7_predict import load
from photdis.stage1 import require_compute
from concurrent.futures import ProcessPoolExecutor
from scipy.special import ndtr
from functools import lru_cache


@lru_cache(maxsize=2)
def cached(path):return load(path)


def worker(task):
    s,method,q,x,ii,ref_sigma=task;p=OUT/'predict'
    if method=='QRF':
        z=cached(p/('QRF_north.npz' if s=='M13' else 'QRF_south.npz'));at={int(k):i for i,k in enumerate(z['ii'])};j=at[int(ii)]
        edge=z['edges'][j];mass=z['mass'][j];width=np.diff(edge);raw=(mass[:,None]*np.clip((x[None]-edge[:-1,None])/width[:,None],0,1)).sum(axis=0)
        conv,lp=uniform_curve(x,edge,mass,ref_sigma)
    else:
        z=cached(p/f'{s}_{method}_{q}.npz');j=int(np.flatnonzero(z['ii']==ii)[0]);mm=z['means'][j];lw=z['logw'][j];sg=z['sigma'][j]
        # Every representable component is retained in the CDF; original finite
        # log weights, including underflowing probabilities, enter log density.
        w=np.exp(lw);use=w>0;raw=np.zeros(len(x));conv=np.zeros(len(x));lp=np.empty(len(x));eff=np.hypot(sg,ref_sigma)
        for a in range(0,len(x),512):
            xx=x[a:a+512];zz=(xx[:,None]-mm[use])/sg;raw[a:a+512]=(ndtr(zz)*w[use]).sum(axis=1)
            conv[a:a+512]=(ndtr((xx[:,None]-mm[use])/eff)*w[use]).sum(axis=1)
            _,lp[a:a+512]=evaluate_mix(xx,mm,lw,eff)
    assert abs(raw[-1]-1)<1e-9 and raw[0]<1e-9
    assert abs(conv[-1]-1)<1e-9 and conv[0]<1e-9 and np.isfinite(lp).all()
    assert np.diff(raw).min()>-1e-10 and np.diff(conv).min()>-1e-10
    return ii,raw,conv,lp,curve_crps(x,conv)


def main():
    require_compute();d=folder('curves');t=load(OUT/'prepare/roster.npz');q=json.loads((OUT/'predict/precision.json').read_text())['chosen'];inputs={'predict':sha(OUT/'predict/complete.json'),'prepare':sha(OUT/'prepare/complete.json')}
    low=0.;high=30.
    for s in C['systems']:
        rs=np.std(5*np.log10(t['scale_'+s]),ddof=1)
        for m in METHODS[:-1]:
            z=load(OUT/'predict'/f'{s}_{m}_{q}.npz');tail=10*np.hypot(z['sigma'].max(),rs);low=min(low,float(z['means'].min()-tail));high=max(high,float(z['means'].max()+tail))
    for side in ('south','north'):
        z=load(OUT/'predict'/f'QRF_{side}.npz');low=min(low,float(z['edges'].min()-2));high=max(high,float(z['edges'].max()+2))
    h=C['mu_grid_step'];x=np.arange(np.floor(low/h)*h,np.ceil(high/h)*h+h/2,h);npz(d/'grid.npz',mu=x)
    status=[]
    for s in C['systems']:
        rows=np.flatnonzero(t['system']==s);rs=np.std(5*np.log10(t['scale_'+s]),ddof=1)
        for m in METHODS:
            if m=='QRF':
                z=load(OUT/'predict'/('QRF_north.npz' if s=='M13' else 'QRF_south.npz'));ii=np.intersect1d(rows,z['ii']);idx=np.array([np.flatnonzero(z['ii']==i)[0] for i in ii]);qs=z['q'][idx];ok=np.ones(len(ii),bool);post=np.full(len(ii),np.nan);chi=np.full(len(ii),np.nan)
            else:
                z=load(OUT/'predict'/f'{s}_{m}_{q}.npz');ii=z['ii'];qs=z['q'];ok=z['support'];post=z['post'];chi=z['chi']
            tasks=[(s,m,q,x,int(i),rs) for i in ii]
            with ProcessPoolExecutor(max_workers=C['workers']) as pool:results=list(pool.map(worker,tasks,chunksize=2))
            np.testing.assert_array_equal([r[0] for r in results],ii)
            npz(d/f'{s}_{m}.npz',ii=ii,q=qs,support=ok,post=post,chi=chi,raw=np.array([r[1] for r in results]),cdf=np.array([r[2] for r in results]),logpdf=np.array([r[3] for r in results]),crps=np.array([r[4] for r in results]))
            for i in rows:
                j=np.flatnonzero(ii==i);status.append(dict(system=s,ls_id=int(t['ls_id'][i]),method=m,supported=bool(len(j) and ok[j[0]]),state=('supported' if ok[j[0]] else 'color_chi2_gt25') if len(j) else 'frozen_QRF_unsupported',chi2=float(chi[j[0]]) if len(j) else None))
            print('Full curves',s,m,len(ii),len(x),flush=True)
    csv_rows(d/'support_status.csv',status);complete(d,inputs,mu_grid=[float(x[0]),float(x[-1]),h],normalized_all_tails=True)


if __name__=='__main__':main()

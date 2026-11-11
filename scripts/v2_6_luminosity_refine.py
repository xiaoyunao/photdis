#!/usr/bin/env python3
"""Same luminosity model/prior/data, finer integration; preserve initial outputs."""
import json
import numpy as np
from scipy.special import logsumexp
import v2_6_luminosity as m
from v2_1_near import ROOT,sha,save_json,atomic_npz,csv_rows
from photdis.population_v2_5 import weighted_quantile
from photdis.stage1 import require_compute


def calibrate(t,rows,poor,ref,sigma,source):
    grid=np.linspace(-1,1,161);scatter=np.linspace(0,.6,25);total=np.linspace(-2,2,321)
    ordinary=rows[~poor[rows]];metal=rows[poor[rows]]
    ol=np.array([m.signed_loglike(ref[ordinary],sigma[ordinary],t['p_corr'][ordinary],t['gaia_parallax_error'][ordinary],offset=z).sum() for z in grid])
    pl=np.empty((len(total),len(scatter)))
    for j,z in enumerate(total):
        for k,s in enumerate(scatter):pl[j,k]=m.signed_loglike(ref[metal],sigma[metal],t['p_corr'][metal],t['gaia_parallax_error'][metal],offset=z,scatter=s).sum()
    a,b,s=np.meshgrid(np.arange(len(grid)),np.arange(len(grid)),np.arange(len(scatter)),indexing='ij')
    ll=ol[a]+pl[a+b,s];w=np.exp(ll-logsumexp(ll)).ravel();parameters=np.column_stack((grid[a.ravel()],grid[b.ravel()],scatter[s.ravel()]))
    atomic_npz(m.D/f'luminosity_posterior_{source}.npz',parameters=parameters,weight=w,identity=t['identity'][rows],TYPE=t['TYPE'][rows])
    mean=w@parameters;qs=[]
    for kind,rr,offset,sc in [('ordinary',ordinary,mean[0],0),('poor',metal,mean[0]+mean[1],mean[2])]:
        args=(ref[rr],sigma[rr],t['p_corr'][rr],t['gaia_parallax_error'][rr])
        delta=m.signed_loglike(*args,offset=offset,scatter=sc,order=32)-m.signed_loglike(*args,offset=offset,scatter=sc,order=64)
        qs.append(dict(source=source,group=kind,n=len(rr),mean_abs_log_difference=float(abs(delta).mean()),max_abs_log_difference=float(abs(delta).max())))
    csv_rows(m.T/f'luminosity_quadrature_{source}.csv',qs)
    out=[]
    for k,name in enumerate(('source_offset','poor_offset','poor_scatter')):
        lo,med,hi=weighted_quantile(parameters[:,k],w);out.append(dict(source=source,parameter=name,mean=float(mean[k]),low=lo,median=med,high=hi,
            cal_poor=len(metal),cal_ordinary=len(ordinary),negative=int(np.sum(t['p_corr'][rows]<0)),weak=int(np.sum(t['p_corr'][rows]/t['gaia_parallax_error'][rows]<5)),boundary=float(w@((abs(parameters[:,0])>.95)|(abs(parameters[:,1])>.95)|(parameters[:,2]>.55)))))
    return parameters,w,out


if __name__=='__main__':
    require_compute();m.D=ROOT/'data/v2_6/luminosity_refined';m.T=m.D/'tables';m.T.mkdir(parents=True,exist_ok=True)
    import shutil
    old=ROOT/'data/v2_6/validation.json';backup=ROOT/'data/v2_6/validation_initial.json'
    if old.exists() and not backup.exists():shutil.copy2(old,backup)
    paths=[ROOT/'scripts/v2_6_luminosity_refine.py',ROOT/'docs/v2_6_luminosity_numerical_amendment.md',ROOT/'data/v2_6/luminosity_freeze.json']
    freeze={str(p.relative_to(ROOT)):sha(p) for p in paths};marker=m.D/'numerical_freeze.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze
    else:save_json(marker,freeze)
    m.calibrate=calibrate;m.main()

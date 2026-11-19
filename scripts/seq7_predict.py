#!/usr/bin/env python3
"""Photometry-only fixed predictions and controlled quadrature convergence."""
from seq7_common import *
from seq7_math import mixture,quantiles,evaluate_mix
from photdis.stage1 import require_compute
from photdis.v3_3 import frozen_qrf
from v3_13_clusters import model_and_history
from stream_sequence import posterior


def load(path):
    with np.load(path) as f:return dict(f)


def sequence(t,q,d):
    audit=[]
    for s in C['systems']:
        ii=np.flatnonzero(t['system']==s);ref=10+5*np.log10(t['ref'][ii]);rs=np.std(5*np.log10(t['scale_'+s]),ddof=1)
        for method in METHODS[:-1]:
            z=load(OUT/'prepare'/f'{s}_{method}_{q}.npz');mm,lw,sigma,chi=mixture(t['m'][ii],t['err'][ii],z['nodes'],z['prior']);qs=quantiles(mm,lw,sigma);ok=chi<=25
            _,score=evaluate_mix(ref,mm,lw,np.sqrt(sigma**2+rs**2));post=np.exp(lw[:,z['post']]).sum(axis=1)
            npz(d/f'{s}_{method}_{q}.npz',ii=ii,means=mm,logw=lw,sigma=sigma,q=qs,support=ok,chi=chi,post=post)
            for phase in ('all','lower_MS','turnoff','other'):
                keep=ok&((t['phase'][ii]==phase) if phase!='all' else True)
                if keep.any():audit.append(dict(system=s,method=method,subdivisions=q,phase=phase,N=int(keep.sum()),median_mu_residual=float(np.median((qs[:,2]-ref)[keep])),mean_logscore=float(np.mean(score[keep]))))
            print('Predicted',s,method,q,int(ok.sum()),flush=True)
    return audit


def main():
    require_compute();d=folder('predict');p=OUT/'prepare';inputs={'prepare':sha(p/'complete.json')};t=load(p/'roster.npz');audit=[];convergence=[]
    # Exact old A/MC replay is separately sealed before the analytic controls.
    legacy=[]
    for s in C['systems']:
        ii=np.flatnonzero((t['system']==s)&t['v5']);z=load(p/f'{s}_A_4.npz');ids=np.where(t['source_id'][ii]>0,t['source_id'][ii],-t['ls_id'][ii]);pred,_=posterior(t['m'][ii],t['err'][ii],ids,(z['nodes'],z['prior']));diff=float(np.max(abs(pred-t['legacy_new'][ii])))
        np.testing.assert_allclose(pred,t['legacy_new'][ii],rtol=2e-6,atol=2e-6);legacy.append(dict(system=s,N=len(ii),legacy_A_MC128_max_distance_difference=diff))
    csv_rows(d/'legacy_A_replay.csv',legacy)
    q=4;audit+=sequence(t,q,d)
    while True:
        fine=q*2;audit+=sequence(t,fine,d);passed=True
        for s in C['systems']:
            for method in METHODS[:-1]:
                a=load(d/f'{s}_{method}_{q}.npz');b=load(d/f'{s}_{method}_{fine}.npz');ii=a['ii'];ok=a['support']&b['support'];ref=10+5*np.log10(t['ref'][ii]);rs=np.std(5*np.log10(t['scale_'+s]),ddof=1)
                _,sa=evaluate_mix(ref,a['means'],a['logw'],np.sqrt(a['sigma']**2+rs**2));_,sb=evaluate_mix(ref,b['means'],b['logw'],np.sqrt(b['sigma']**2+rs**2))
                for phase in ('all','lower_MS','turnoff','other'):
                    use=ok&((t['phase'][ii]==phase) if phase!='all' else True)
                    if not use.any():continue
                    change=float(np.median(b['q'][use,2])-np.median(a['q'][use,2]));score=float(np.mean(sb[use]-sa[use]));p98=float(np.quantile(abs(b['q'][use,2]-a['q'][use,2]),.98));good=abs(change)<=.005 and abs(score)<=.005 and p98<=.01
                    convergence.append(dict(system=s,method=method,phase=phase,coarse=q,fine=fine,N=int(use.sum()),median_shift=change,mean_logscore_shift=score,p98_star_shift=p98,support_changed=int(sum(a['support']!=b['support'])),pass_precision=good));passed&=good
        csv_rows(d/'convergence.csv',convergence)
        if passed:chosen=fine;break
        if fine>=C['max_subdivisions']:chosen=fine;break
        q=fine
    csv_rows(d/'quadrature_metrics.csv',audit)
    qrf_replay=[]
    for code,side in enumerate(('south','north')):
        model,offset,train=model_and_history(side,inputs);ii=np.flatnonzero((t['native']==code)&np.isfinite(t['q']).all(axis=1)&(t['q']>0).all(axis=1));assert not np.isin(t['qrf_identity'][ii],train).any()
        edges=[];mass=[];quant=[]
        for start in range(0,len(ii),64):
            use=ii[start:start+64];z=frozen_qrf(model,t['qrf_mag'][use],t['qrf_err'][use],t['qrf_identity'][use]);assert z['supported'].all();np.testing.assert_allclose(z['distance']*10**(offset/5),t['q'][use],atol=2e-6,rtol=2e-6)
            edges.extend(z['mu_edges']+offset);mass.extend(z['mass']);quant.extend(z['distance']*10**(offset/5))
        npz(d/f'QRF_{side}.npz',ii=ii,edges=np.array(edges),mass=np.array(mass),q=10+5*np.log10(quant));qrf_replay.append(dict(side=side,N=len(ii),max_distance_error=float(np.max(abs(np.array(quant)-t['q'][ii]))),original_float64=True))
    csv_rows(d/'QRF_replay.csv',qrf_replay);save_json(d/'precision.json',dict(chosen=chosen,criterion_pass=bool(passed),max_resolution=C['max_subdivisions'],all_prior_trials_retained=True))
    complete(d,inputs,chosen_subdivisions=chosen,precision_pass=bool(passed));print('Prediction complete',chosen,passed,flush=True)


if __name__=='__main__':main()

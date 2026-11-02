#!/usr/bin/env python3
"""Frozen external predictions after independent identity/quality inventory."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
from photdis.kgiant_distance import predict_mixture
from photdis.giant_kernel import mixture_quantiles
from photdis.stage1 import require_compute,write_json
from kgiant_distance_inventory import sha

OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/kgiant_external_v1')

def main():
    require_compute()
    if (OUT/'predictions.json').exists():raise FileExistsError(OUT/'predictions.json')
    inv=json.load(open(OUT/'inventory.json')); q=np.load('data/kgiant_distance_v1/population.npz')
    bands=np.column_stack([q[k] for k in ('g0','r0','z0','w10')]);errors=np.column_stack([q[k] for k in ('sigma_g','sigma_r','sigma_z','sigma_w1')])
    refids=[int(r['source_id']) for r in inv['rows'] if r.get('source_id','0')!='0']
    rng=np.random.default_rng(20261102)
    scales={c:dict(delta_ebv=rng.normal(0,.01,256),delta_mu=rng.normal(0,sm,256)) for c,sm in [('Pal12',.1),('Terzan7',5/np.log(10)*.6/25.3)]}
    np.savez_compressed(OUT/'shared_scale_draws.npz',**{c+'_'+k:v for c,d in scales.items() for k,v in d.items()})
    trainsets={};records=[];audit=[]
    for teacher in ('spdist','specdis'):
        good=q['w1_qualified']&q[teacher+'_valid']&np.isfinite(q[teacher+'_sigma_mu'])&(q[teacher+'_sigma_mu']>0)&(q['cluster_group']==0)&~np.isin(q['source_id'],refids)
        ix=np.flatnonzero(good); trainsets[teacher]=ix
        assert len(ix)==(77 if teacher=='spdist' else 258)
    # Materialize training identity provenance before any reference comparison.
    write_json(OUT/'training.json',{t:dict(n=len(ix),source_ids=[str(i) for i in q['source_id'][ix]],
        cluster_groups=np.unique(q['cluster_group'][ix]).tolist()) for t,ix in trainsets.items()})
    for r in inv['rows']:
        identity_ok=r['cluster']=='Terzan7' and r['identity_status']=='unique_position_match'
        # Pal12 currently lacks an independently identified literature temperature.
        type_ok=identity_ok and r.get('nominal_K',False)
        reasons=[]
        if not identity_ok:reasons.append('external_temperature_identity_unverified' if r['cluster']=='Pal12' else 'external_identity_ambiguous_or_missing')
        elif not type_ok:reasons.append('temperature_outside_K_scope')
        if 'grz_quality' not in r:reasons.append('grz_measurement_quality_not_assigned')
        elif not r['grz_quality']:reasons.append('grz_measurement_quality_failed')
        entry=dict(cluster=r['cluster'],name=r['name'],eligible=not reasons,reasons=reasons,support={})
        ph=r.get('photometry',{})
        qb=np.array([ph.get(k) if ph.get(k) is not None else np.nan for k in ('g0','r0','z0','w10')])
        qe=np.array([ph.get(k) if ph.get(k) is not None else np.nan for k in ('sigma_g','sigma_r','sigma_z','sigma_w1')])
        for teacher,ix in trainsets.items():
            for model,dim in [('prior',0),('A_grz',2),('B_grzW1',3)]:
                nb=dim+1 if dim else 1
                if not np.all(np.isfinite(qb[:nb])):continue
                if dim:
                    tc=bands[ix,:dim]-bands[ix,1:dim+1];qc=qb[:dim]-qb[1:dim+1]
                    entry['support'][teacher+'_'+model]=dict(outside_box=bool(np.any((qc<tc.min(0))|(qc>tc.max(0)))),
                        query_colors=qc.tolist(),train_min=tc.min(0).tolist(),train_max=tc.max(0).tolist())
                if reasons or (dim==3 and not r['w1_quality']):continue
                tm=10+5*np.log10(q[teacher+'_dist_kpc'][ix]);ts=q[teacher+'_sigma_mu'][ix]
                pred=lambda b:predict_mixture(bands[ix],errors[ix],tm,ts,b,qe,dimension=dim)
                baseline=pred(qb); mu_ref=16.3 if r['cluster']=='Pal12' else 17.05
                name=r['cluster']+'_'+r['name']+'_'+teacher+'_'+model
                np.savez_compressed(OUT/(name+'.npz'),**baseline,train_source_id=q['source_id'][ix])
                quant=mixture_quantiles(baseline['means'],baseline['sigmas'],baseline['weights'])
                # Transmission-derived coefficients in the catalog EBV coordinate.
                obs=np.array([ph.get(k) if ph.get(k) is not None else np.nan for k in ('g','r','z','w1')])
                coeff=(obs-qb)/r['ebv']
                perturb=[]
                for de in [-.01,0.,.01]:
                    p=pred(qb-coeff*de);pq=mixture_quantiles(p['means'],p['sigmas'],p['weights'])
                    perturb.append(dict(delta_ebv=de,quantiles_mu=pq.tolist()))
                mm=[];ss=[];ww=[]
                for de,dm in zip(scales[r['cluster']]['delta_ebv'],scales[r['cluster']]['delta_mu']):
                    p=pred(qb-coeff*de);mm.append(p['means']-mu_ref-dm);ss.append(p['sigmas']);ww.append(p['weights']/256)
                m,s,w=np.concatenate(mm),np.concatenate(ss),np.concatenate(ww)
                assert np.isclose(w.sum(),1)
                np.savez_compressed(OUT/(name+'_scale_delta.npz'),means=m,sigmas=s,weights=w)
                from scipy.special import ndtr,logsumexp
                grid=np.linspace(float(np.min(baseline['means']-5*baseline['sigmas'])),float(np.max(baseline['means']+5*baseline['sigmas'])),501)
                standardized=(grid[:,None]-baseline['means'])/baseline['sigmas']
                pdf=np.exp(logsumexp(np.log(baseline['weights'])[None,:]-.5*standardized**2-np.log(baseline['sigmas'])[None,:]-.5*np.log(2*np.pi),axis=1))
                cdf=ndtr(standardized)@baseline['weights']
                records.append(dict(cluster=r['cluster'],name=r['name'],teacher=teacher,model=model,
                    reference_mu=mu_ref,quantiles_mu=quant.tolist(),distance_kpc_quantiles=(10**((quant-10)/5)).tolist(),
                    delta_mu_median=float(quant[2]-mu_ref),outside_training_box=baseline['outside_training_box'],
                    effective_n=baseline['effective_n'],perturbations=perturb,
                    shared_scale_delta_quantiles=mixture_quantiles(m,s,w).tolist(),
                    spdist_temperature_extrapolation=teacher=='spdist' and r['teff']<4800,
                    distribution_grid_mu=grid.tolist(),distribution_pdf=pdf.tolist(),distribution_cdf=cdf.tolist()))
        audit.append(entry)
    write_json(OUT/'predictions.json',dict(rows=records,audit=audit,training_n={t:len(v) for t,v in trainsets.items()},
        source_sha256={str(p):sha(p) for p in [Path(__file__),OUT/'inventory.json',OUT/'training.json',OUT/'shared_scale_draws.npz',
        Path('data/kgiant_distance_v1/population.npz'),Path('src/photdis/kgiant_distance.py'),Path('docs/kgiant_external_protocol_v1.md')]},
        scope='six-target scale/failure check only; no population precision or coverage calibration'))
    print('Prediction rows',len(records),'; target audit',audit,flush=True)

if __name__=='__main__':main()

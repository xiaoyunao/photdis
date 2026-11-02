#!/usr/bin/env python3
"""Score the frozen member-first inventory, without changing teacher models."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
from scipy.special import ndtr
from photdis.kgiant_distance import predict_mixture
from photdis.giant_kernel import mixture_quantiles
from photdis.stage1 import require_compute,write_json
from kgiant_distance_inventory import sha
from kgiant_reference_batch import OUT,REF,BANDS,ERRORS

def main():
    require_compute()
    if (OUT/'predictions.json').exists():raise FileExistsError(OUT/'predictions.json')
    inv=json.load(open(OUT/'inventory.json'));refs={r['name']:r for r in json.load(open(REF))['rows']}
    q=np.load('data/kgiant_distance_v1/population.npz');excluded=np.load(OUT/'all_reference_member_ids.npz')
    ids=np.r_[excluded['source_id'],excluded['ambiguous_ids']]
    bands=np.column_stack([q[k] for k in BANDS]);err=np.column_stack([q[k] for k in ERRORS]);train={}
    for teacher in ('spdist','specdis'):
        base=q['w1_qualified']&q[teacher+'_valid']&np.isfinite(q[teacher+'_sigma_mu'])&(q[teacher+'_sigma_mu']>0)&(q['cluster_group']==0)
        mask=base&~np.isin(q['source_id'],ids)
        train[teacher]=np.flatnonzero(mask)
    write_json(OUT/'training.json',{t:dict(n=len(ix),source_ids=[str(s) for s in q['source_id'][ix]],reference_overlap=0) for t,ix in train.items()})
    rows=[];checks=[]
    for r in inv['rows']:
        if not r['eligible']:continue
        m=r['selected_measurement'];ph=m['photometry'];b=np.array([ph[k] if ph[k] is not None else np.nan for k in BANDS]);e=np.array([ph[k] if ph[k] is not None else np.nan for k in ERRORS])
        ref=refs[r['reference_name']];refmu=ref['mu'];coeff=(np.array([ph[k] if ph[k] is not None else np.nan for k in ('g','r','z','w1')])-b)/m['ebv'] if m['ebv']>0 else np.zeros(4)
        for teacher,ix in train.items():
            assert int(r['source_id']) not in q['source_id'][ix]
            tm=10+5*np.log10(q[teacher+'_dist_kpc'][ix]);ts=q[teacher+'_sigma_mu'][ix]
            for model,dim in [('prior',0),('A_grz',2),('B_grzW1',3)]:
                if dim==3 and not m['w1_quality']:continue
                p=predict_mixture(bands[ix],err[ix],tm,ts,b,e,dimension=dim)
                quant=mixture_quantiles(p['means'],p['sigmas'],p['weights']);prefix=r['source_id']+'_'+teacher+'_'+model
                np.savez_compressed(OUT/(prefix+'.npz'),**p,train_source_id=q['source_id'][ix])
                grid=np.linspace(np.min(p['means']-5*p['sigmas']),np.max(p['means']+5*p['sigmas']),201)
                v=(grid[:,None]-p['means'])/p['sigmas'];cdf=ndtr(v)@p['weights'];pdf=(np.exp(-v*v/2)/np.sqrt(2*np.pi)/p['sigmas'])@p['weights']
                assert np.isclose(p['weights'].sum(),1) and np.all(np.diff(cdf)>=-1e-12)
                assert np.all(np.diff(quant)>0) and cdf[0]<1e-5 and cdf[-1]>1-1e-5
                pert=[]
                for delta in (-.01,.01):
                    pp=predict_mixture(bands[ix],err[ix],tm,ts,b-coeff*delta,e,dimension=dim)
                    qq=mixture_quantiles(pp['means'],pp['sigmas'],pp['weights'])
                    pert.append(dict(delta_catalog_ebv=delta,delta_mu_median=float(qq[2]-quant[2])))
                support=[]
                if dim:
                    colors=b[:dim]-b[1:dim+1];tc=bands[ix,:dim]-bands[ix,1:dim+1]
                    for name,c,lo,hi in zip(('gr','rz','zw1'),colors,tc.min(0),tc.max(0)):
                        support.append(dict(color=name,value=float(c),train_min=float(lo),train_max=float(hi),
                           signed_excursion=float(c-lo if c<lo else c-hi if c>hi else 0)))
                rows.append(dict(source_id=r['source_id'],name=r['name'],system=r['member_system'],reference_name=r['reference_name'],
                  historical_diagnostic=r['historical_diagnostic'],teacher=teacher,model=model,quantiles_mu=quant.tolist(),
                  distance_kpc_quantiles=(10**((quant-10)/5)).tolist(),delta_mu_median=float(quant[2]-refmu),reference_mu=refmu,
                  reference_mu_minus=ref['mu_minus'],reference_mu_plus=ref['mu_plus'],
                  residual_reference_only_bounds=[float(quant[2]-refmu-ref['mu_plus']),float(quant[2]-refmu+ref['mu_minus'])],
                  reference_method_centers={k:float(quant[2]-10-5*np.log10(val['distance_kpc'])) for k,val in ref['methods'].items() if val},
                  effective_n=p['effective_n'],outside_training_box=p['outside_training_box'],support=support,
                  training_teff_minmax=[float(q['desi_teff'][ix].min()),float(q['desi_teff'][ix].max())],
                  external_teff=r['teff'],temperature_scale_caveat=r['type_source'],
                  old_latitude_scope=r['old_latitude_scope'],old_dust_scope=r['old_dust_scope'],
                  perturbations=pert,distribution_grid_mu=grid.tolist(),distribution_pdf=pdf.tolist(),distribution_cdf=cdf.tolist()))
    # Descriptive star-to-star centers, never products of individually marginalized PDFs.
    summaries=[]
    for system in sorted({r['system'] for r in rows}):
        for teacher in train:
            for model in ('prior','A_grz','B_grzW1'):
                rr=[r for r in rows if r['system']==system and r['teacher']==teacher and r['model']==model]
                if not rr:continue
                delta=np.array([r['delta_mu_median'] for r in rr]);median=float(np.median(delta))
                summaries.append(dict(system=system,teacher=teacher,model=model,n_stars=len(rr),median_delta=median,
                    star_to_star_nmad=float(1.4826*np.median(abs(delta-median))),
                    reference_only_center_bounds=[median-rr[0]['reference_mu_plus'],median+rr[0]['reference_mu_minus']],
                    common_ebv_center_shifts=[float(np.median([r['delta_mu_median']+r['perturbations'][j]['delta_mu_median'] for r in rr])-median) for j in (0,1)]))
    pairs=[]
    for rb in [r for r in rows if r['model']=='B_grzW1']:
        ra=next(r for r in rows if r['source_id']==rb['source_id'] and r['teacher']==rb['teacher'] and r['model']=='A_grz')
        pairs.append(dict(source_id=rb['source_id'],system=rb['system'],teacher=rb['teacher'],B_minus_A=rb['quantiles_mu'][2]-ra['quantiles_mu'][2],
             absolute_reference_error_change=abs(rb['delta_mu_median'])-abs(ra['delta_mu_median'])))
    paths=[OUT/'inventory.json',OUT/'training.json',OUT/'all_reference_member_ids.npz',REF,Path(__file__),Path('src/photdis/kgiant_distance.py'),Path('data/kgiant_distance_v1/population.npz')]
    write_json(OUT/'predictions.json',dict(rows=rows,system_summaries=summaries,paired=pairs,
         source_sha256={str(p):sha(p) for p in paths},n_stars=len({r['source_id'] for r in rows}),n_systems=len({r['system'] for r in rows}),
         caveat='Empirical teacher-scale mixtures, descriptive centers; no field-giant calibration or joint distance posterior'))
    print('Scored',len(rows),'configurations',len({r['source_id'] for r in rows}),'stars',flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Matched radial posterior recovery; all optima, chains and failures retained."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.sampling import hmc,chain_diagnostics
from photdis.v3.population import SharedCounts
from photdis.v3.likelihood import trapezoid_weights,ContinuousPosterior
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.metrics import distance_summary,observed_parallax
from photdis.v3_1.radial import RadialPopulation
from photdis.v3_1.radial_products import bin_integrals,generator_joint_parameters

_MODEL=_STARTS=_OUT=_ARGS=_INPUTS=None


def sample_chain(i):
    path=_OUT/f'chain_{i}.joblib'
    if path.exists():return joblib.load(path)
    initial=_STARTS[i%len(_STARTS)]
    record=hmc(_MODEL.objective,initial['map'],initial['metric'],draws=_ARGS.draws,warmup=_ARGS.warmup,
        target_acceptance=.9,seed=313000+_ARGS.catalog*10+i+(10000 if _ARGS.arm=='local' else 0))
    record['start_objective']=initial['objective'];joblib.dump(record,path)
    return record


def main():
    global _MODEL,_STARTS,_OUT,_ARGS,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--catalog',type=int,required=True)
    p.add_argument('--arm',choices=['broad','local'],required=True);p.add_argument('--step',type=float,default=.02)
    p.add_argument('--draws',type=int,default=512);p.add_argument('--warmup',type=int,default=512)
    a=_ARGS=p.parse_args();start=time.monotonic();root=ROOT/'data/v3_1'
    folder=root/'radial_likelihood'/f'catalog_{a.catalog:03}_step{a.step:g}';curves=folder/'curves.npz'
    complete=json.loads((folder/'complete.json').read_text());assert complete['selection_passed']
    assert sha(curves)==complete['outputs'][curves.name]
    paths=[curves,folder/'complete.json',root/'radial_catalogs/truth.npz',root/'radial_catalogs/catalogs.json',
        ROOT/'docs/v3_1_radial_protocol.md',ROOT/'scripts/v3_1_radial_recovery.py',ROOT/'src/photdis/v3_1/radial.py',
        ROOT/'src/photdis/v3_1/radial_products.py',ROOT/'src/photdis/v3/sampling.py',ROOT/'src/photdis/v3/population.py']
    inputs=_INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};inputs.update(vars(a))
    _OUT=root/'radial_recovery'/f'catalog_{a.catalog:03}_{a.arm}_step{a.step:g}_s{a.draws}_w{a.warmup}';_OUT.mkdir(parents=True,exist_ok=True)
    freeze=_OUT/'inputs.json';marker=_OUT/'complete.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    if marker.exists():
        saved=json.loads(marker.read_text());assert saved['inputs']==inputs
        for name,digest in saved['outputs'].items():assert sha(_OUT/name)==digest
        return
    data=load(curves);grid=data['mu'];assert (data['TYPE']=='PSF').all()
    offset=data['log_likelihood'].max(axis=(1,2));scaled=np.exp(data['log_likelihood']-offset[:,None,None])
    _MODEL=RadialPopulation(grid,scaled,data['entry'],data['class_frequency'],local=a.arm=='local')
    map_path=_OUT/'map_optima.joblib'
    if map_path.exists():optima=joblib.load(map_path)
    else:optima=_MODEL.fit();joblib.dump(optima,map_path)
    valid=[o for o in optima if np.isfinite(o['objective']) and o['gradient_max']<1e-3]
    if not valid:raise RuntimeError('No accepted MAP; all failed optima retained')
    _STARTS=[]
    for o in valid:
        if o['objective']>valid[0]['objective']+10:continue
        if all(np.linalg.norm(o['map']-x['map'])>.05 for x in _STARTS):_STARTS.append(o)
    _STARTS=_STARTS[:4]
    print('RADIAL MAP',a.catalog,a.arm,valid[0]['objective'],len(_STARTS),'competitive optima',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('fork')) as pool:
        chains=list(pool.map(sample_chain,range(4)))
    samples=np.array([r['samples'] for r in chains]);diagnostics=chain_diagnostics(samples)
    z=samples.transpose(1,0,2).reshape(-1,samples.shape[-1]);s=len(z);n=len(scaled)
    atomic_npz(_OUT/'chains.npz',samples=samples,rhat=diagnostics['rhat'],ess=diagnostics['ess'])
    ratio=np.zeros_like(scaled);probability=np.empty((s,n,18));parameters=np.empty((s,18));structure=np.empty((s,5))
    w=trapezoid_weights(grid)
    for j,value in enumerate(z):
        prior=_MODEL.prior_density(value);unnormalized=scaled*prior[None]
        norm=np.einsum('ncm,m->n',unnormalized,w)
        ratio+=prior[None]/norm[:,None,None]/s
        probability[j]=(bin_integrals(unnormalized,grid)/norm[:,None,None]).reshape(n,18)
        parameters[j]=bin_integrals(prior,grid).reshape(18)
        _,eta,location,width=_MODEL.decode(value);structure[j]=np.r_[eta,location,width]
    counts=SharedCounts(np.arange(s),18,seed=314000+a.catalog+(10000 if a.arm=='local' else 0))
    marginal=counts.add(probability,np.arange(s)).reshape(n,3,6);result=counts.result()
    atomic_npz(_OUT/'counts.npz',**result)
    lp=logsumexp(data['log_likelihood']-offset[:,None,None]+np.log(ratio),axis=1)
    normalization=logsumexp(lp+np.log(w)[None],axis=1);lp-=normalization[:,None]
    quantiles=np.array([ContinuousPosterior(grid,np.exp(x)[None],0.,{}).quantiles() for x in lp])
    direct_joint=bin_integrals(scaled*ratio,grid)
    joint_delta=float(abs(direct_joint-marginal).max())
    # Only after all posterior fitting, read catalog truth for evaluation.
    truth=load(root/'radial_catalogs/truth.npz');use=truth['catalog']==a.catalog;truth={k:v[use] for k,v in truth.items()}
    np.testing.assert_array_equal(truth['identity'],data['identity'])
    catalog=json.loads((root/'radial_catalogs/catalogs.json').read_text())[a.catalog]
    parameter_truth=generator_joint_parameters(catalog).ravel()
    bins=np.searchsorted(DISTANCE_MU_BOUNDARIES,truth['mu'],side='right')
    realized=np.bincount(truth['class_label']*6+bins,minlength=18)
    rows=[]
    for estimand,draws,true in [('realized_selected_count',result['count_draws'],realized),('pre_entry_population_fraction',parameters,parameter_truth)]:
        for b in range(18):
            lo,hi=np.quantile(draws[:,b],[.025,.975])
            mean=result['mean'][b] if estimand=='realized_selected_count' else draws[:,b].mean()
            rows.append(dict(estimand=estimand,class_index=b//6,distance_bin=b%6,truth=float(true[b]),mean=float(mean),
                lower95=float(lo),upper95=float(hi),covered95=bool(lo<=true[b]<=hi)))
    csv_rows(_OUT/'recovery.csv',rows)
    metrics=[]
    for name,mask in [('all',np.ones(n,bool)),('2_5kpc',bins==1)]+[(f'class_{c}',truth['class_label']==c) for c in range(3)]:
        metrics.append(dict(group=name,**distance_summary(quantiles[mask],truth['mu'][mask])))
    csv_rows(_OUT/'distance.csv',metrics)
    scores=np.array([np.interp(x,grid,y) for x,y in zip(truth['mu'],lp)])
    gaia=[observed_parallax(grid,y,p,e) for y,p,e in zip(lp,truth['parallax'],truth['parallax_error'])]
    atomic_npz(_OUT/'per_star.npz',identity=data['identity'],TYPE=data['TYPE'],mu=quantiles,class_distance=marginal,
        grid=grid,logpdf=lp,truth_mu=truth['mu'],truth_class=truth['class_label'],logscore=scores,
        gaia_pit=[x['pit'] for x in gaia],gaia_logscore=[x['log_score'] for x in gaia])
    atomic_npz(_OUT/'population_parameters.npz',joint_fractions=parameters,truth_joint_fractions=parameter_truth,structure=structure)
    structure_result={}
    if a.arm=='local':
        true_location=catalog.get('true_location_mu',10+5*np.log10(4.) if catalog['scenario']=='narrow_4kpc' else None)
        true_width=catalog.get('true_width_mu',.08 if catalog['scenario']=='narrow_4kpc' else None)
        for name,column,true in [('location_mu',3,true_location),('width_mu',4,true_width)]:
            lo,med,hi=np.quantile(structure[:,column],[.025,.5,.975])
            structure_result[name]=dict(lower95=lo,median=med,upper95=hi,truth=true,covered95=None if true is None else bool(lo<=true<=hi))
        structure_result['class_amplitude_median']=np.median(structure[:,:3],axis=0)
    sampler=diagnostics['max_rhat']<1.05 and diagnostics['minimum_ess']>=100 and not sum(r['divergent'] for r in chains)
    passed=sampler and joint_delta<1e-8 and abs(normalization).max()<1e-8 and result['marginal_half_delta']<.005
    save_json(marker,dict(inputs=inputs,outputs={p.name:sha(p) for p in _OUT.iterdir() if p.suffix in ['.npz','.joblib','.csv']},
        job=os.getenv('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-start,catalog=a.catalog,scenario=catalog['scenario'],arm=a.arm,
        map_optima=len(optima),competitive_chain_starts=len(_STARTS),max_rhat=diagnostics['max_rhat'],minimum_ess=diagnostics['minimum_ess'],
        divergent=sum(r['divergent'] for r in chains),sampler_passed=bool(sampler),passed=bool(passed),
        joint_delta=joint_delta,max_log_normalization_delta=float(abs(normalization).max()),marginal_half_delta=result['marginal_half_delta'],
        structure=structure_result,scope='Fixed nominal SED/zero tilt radial comparison. Pre-entry parameter fractions differ from realized selected counts. Failed chains/results retained; numerical acceptance is not scientific calibration.'))
    print('RADIAL RECOVERY COMPLETE',a.catalog,a.arm,bool(passed),diagnostics['max_rhat'],flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Observed-count recovery for one fully refitted shared response realization."""
import argparse
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.compression import unpack
from photdis.v3.target import component_response,joint_coefficients,joint_draw_probabilities
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.sampling import hmc,chain_diagnostics
from photdis.v3.selection_table import SelectionTable
from photdis.v3.selection import effective_errors
from photdis.v3.population import SharedCounts
from photdis.v3.marginal import marginal_coefficients
from photdis.v3.marginal_cdf import cdf_product
from photdis.v3.evidence import bridge_integral
from photdis.v3.systematics import calibration_vectors
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--draw',type=int,required=True)
    p.add_argument('--catalog',type=int,required=True);p.add_argument('--draws',type=int,default=2048)
    p.add_argument('--warmup',type=int,default=512);a=p.parse_args();start=time.monotonic()
    pred=D/'shared_predictions/simulations_refit_0'/f'draw_{a.draw:02}'
    model_path=D/'uncertainty_refit'/f'draw_{a.draw:02}/model.joblib'
    selection=D/'shared_selection'/f'refit_{a.draw:02}_p512_q256/table.npz'
    if a.draw==16:selection=D/'shared_selection/refit_16_p256/table.npz'
    srecord=json.loads(selection.with_name('complete.json').read_text());assert srecord['passed'] and sha(selection)==srecord['sha256']
    mrecord=json.loads(model_path.with_name('complete.json').read_text());assert mrecord['converged'] and sha(model_path)==mrecord['sha256']
    assert (pred/'complete.json').exists()
    source_path=D/'cohorts/simulations.npz';truth_path=D/'simulations/truth.npz';shared_path=D/'uncertainty/shared_draws.npz'
    paths=[source_path,truth_path,model_path,selection,pred/'complete.json',shared_path,ROOT/'scripts/v3_0_shared_recovery.py']
    paths += [ROOT/'src/photdis/v3'/x for x in ('target.py','fast_hierarchy.py','hierarchy.py','sampling.py','evidence.py','marginal.py','marginal_cdf.py','population.py','systematics.py')]
    inputs={str(x.relative_to(ROOT)):sha(x) for x in paths};inputs.update(vars(a))
    out=D/'shared_recovery'/f'refit_{a.draw:02}'/f'catalog_{a.catalog:03}_s{a.draws}';out.mkdir(parents=True,exist_ok=True)
    if (out/'inputs.json').exists():assert json.loads((out/'inputs.json').read_text())==inputs
    else:save_json(out/'inputs.json',inputs)
    if (out/'complete.json').exists():
        record=json.loads((out/'complete.json').read_text());assert record['inputs']==inputs
        for n,h in record['outputs'].items():assert sha(out/n)==h
        return
    source=load(source_path);truth=load(truth_path);use=np.flatnonzero(truth['catalog']==a.catalog);assert len(use)==128
    phot={k:v[use] for k,v in source.items()};truth={k:v[use] for k,v in truth.items()}
    assert (phot['TYPE']=='PSF').all();compact=[]
    for first in range(int(use[0]),int(use[-1])+1,32):
        path=pred/f'chunk_{first:08}.npz';meta=json.loads(path.with_suffix('.json').read_text())
        summary=load(path);joint=path.with_name(path.stem+'_joint.npz');assert sha(path)==meta['summary_sha256'] and sha(joint)==meta['joint_sha256']
        np.testing.assert_array_equal(summary['identity'],source['identity'][first:first+len(summary['identity'])])
        payload=load(joint);compact.extend(unpack(payload,j) for j in range(len(summary['identity'])))
    model=inference_model(joblib.load(model_path));table=SelectionTable(**load(selection))
    response=np.array([component_response(r,model) for r in compact])
    calibration,_=calibration_vectors(phot,load(shared_path),a.draw)
    alpha,audit=table.evaluate(effective_errors(phot)*10**(.4*calibration[:,[0,1,3]]));assert audit['inside_profile_hull'].all()
    hierarchy=FastFieldPopulation(response.sum(axis=3),alpha,model,phot['field'])
    if (out/'map.joblib').exists():fit=joblib.load(out/'map.joblib')
    else:fit=hierarchy.fit();joblib.dump(fit,out/'map.joblib')
    records=[]
    for i in range(4):
        path=out/f'chain_{i}.joblib'
        if path.exists():record=joblib.load(path)
        else:
            record=hmc(hierarchy.objective,fit['map'],fit['metric'],draws=a.draws,warmup=a.warmup,
                seed=310000+a.draw*10000+a.catalog*10+i,target_acceptance=.9);joblib.dump(record,path)
        records.append(record)
    samples=np.array([r['samples'] for r in records]);diagnostic=chain_diagnostics(samples)
    z=samples.transpose(1,0,2).reshape(-1,samples.shape[-1]);decoded=[hierarchy.decode(x)[:2] for x in z]
    assert len(hierarchy.fields)==1
    theta=np.array([x[0][0] for x in decoded]);beta=np.array([x[1] for x in decoded])
    coef=joint_coefficients(model,theta,beta);prob=joint_draw_probabilities(response,coef)
    counts=SharedCounts(np.arange(len(z)),18,seed=311000+a.catalog+a.draw*1000)
    mean=counts.add(prob.reshape(len(z),128,18),np.arange(len(z)));atomic_npz(out/'counts.npz',**counts.result())
    avg,_=marginal_coefficients(response,coef)
    curves=[cdf_product(r,model,c) for r,c in zip(compact,avg)]
    offset=np.r_[0,np.cumsum([len(r['grid']) for r in curves])]
    atomic_npz(out/'per_star.npz',identity=phot['identity'],TYPE=phot['TYPE'],probability=mean.reshape(128,3,6),
        half_a=prob[::2].mean(axis=0),half_b=prob[1::2].mean(axis=0),truth_mu=truth['mu'],truth_class=truth['class_label'],
        mu=np.array([r['mu'] for r in curves]),cdf_grid=np.concatenate([r['grid'] for r in curves]),
        class_cdf=np.concatenate([r['class_cdf'] for r in curves],axis=1),cdf_offset=offset,
        omitted_bound=np.array([r['omitted_posterior_bound'] for r in curves]))
    def logdensity(x):return -hierarchy.objective(x)[0]-.5*hierarchy.dimension*np.log(2*np.pi)
    bridge=bridge_integral(logdensity,z,proposal_draws=2048,seed=312000+a.draw*1000+a.catalog)
    bridge.pop('log_ratio_posterior');bridge.pop('log_ratio_proposal')
    bridge['log_evidence']=bridge['log_integral']+sum(r['source_log_evidence'] for r in compact)
    numerical=diagnostic['max_rhat']<1.05 and diagnostic['minimum_ess']>=100 and not sum(r['divergent'] for r in records)
    passed=numerical and bridge['fixed_point_converged'] and bridge['proposal_bridge_ess']>=100 and bridge['half_log_integral_difference']<.2
    save_json(out/'complete.json',dict(inputs=inputs,catalog=a.catalog,global_response_draw=a.draw,scenario=str(truth['scenario'][0]),
        passed=bool(passed),sampler_passed=bool(numerical),max_rhat=diagnostic['max_rhat'],minimum_ess=diagnostic['minimum_ess'],
        conditional_half_delta=counts.result()['marginal_half_delta'],max_omitted_bound=max(r['omitted_posterior_bound'] for r in curves),
        bridge=bridge,outputs={p.name:sha(p) for p in out.glob('*.npz')},elapsed_seconds=time.monotonic()-start,job=os.getenv('SLURM_JOB_ID')))
    print('SHARED RECOVERY',a.draw,a.catalog,passed,flush=True)


if __name__=='__main__':main()

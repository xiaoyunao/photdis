#!/usr/bin/env python3
"""Matched response12 reference-versus-shared-target-prior diagnostic."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
import joblib
import numpy as np
from scipy.special import softmax
from scipy.stats import norm
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from v3_0_response import observation
import v3_0_evaluate as evaluator
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.v3.model import inference_model
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.inference import kernel_loglike
from photdis.v3.systematics import calibration_vectors
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.metrics import observed_parallax,normal_reference_score
from photdis.v3_1.shared_prior import geographic_fields,marginalize_native
from photdis.v3_1.diagnostics import infer_control,mode_attribution

_TABLE=_EVIDENCE=_MODEL=_CAL=_THETA=_BETA=_WEIGHTS=_DRAW_IDS=_FIELDS=_EXISTING=_OVERLAP=_VIEWS=_SOURCE=_OUT=_INPUTS=None


def one(row):
    identity=int(_TABLE['identity'][row]);marker=_OUT/f'object_{identity}.json';pdf=marker.with_suffix('.npz')
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS and sha(pdf)==r['pdf_sha256'];return r
    source=_SOURCE/f'object_{identity}.json';old=json.loads(source.read_text())
    reference=next(r for r in old['results'] if r['control']=='response12')
    obs=observation(_TABLE,row);base=infer_control(_MODEL,obs,calibration_mag=_CAL[row])
    grid=np.unique(np.r_[base['posterior'].mu,np.linspace(-20,50,1401),DISTANCE_MU_BOUNDARIES])
    results=[];arrays={};field=int(_FIELDS[row]);theta=_THETA[field]
    for name,g in [('coarse',grid),('fine',np.unique(np.r_[grid,(grid[1:]+grid[:-1])/2]))]:
        likelihood,_=kernel_loglike(_MODEL,obs,g,calibration_mag=_CAL[row])
        r=marginalize_native(_MODEL,g,likelihood,theta,_BETA,_WEIGHTS)
        p=r['posterior'];q=p.quantiles();results.append((r,q))
        arrays.update({name+'_grid':g,name+'_logpdf':r['full_logpdf'],name+'_class_distance':r['class_distance'],name+'_mu':q})
    r,q=results[-1];p=r['posterior'];lp=r['full_logpdf']
    delta=float(abs(r['class_distance']-results[0][0]['class_distance']).max());mu_delta=float(abs(q-results[0][1]).max())
    passed=bool(delta<.003 and mu_delta<.03 and abs(r['log_normalization'])<1e-8 and r['joint_delta']<1e-8 and r['marginal_half_delta']<.005)
    record=dict(reference);record.update(control='response12_target',prior='shared_target_refit12_s4096_native',
        mu=q.tolist(),qnear=float(r['class_distance'][:,:3].sum()),class_distance=r['class_distance'].tolist(),
        modes=mode_attribution(p,len(_MODEL.weights),getattr(_MODEL,'parent_component',None)),
        target_field=field,existing_target_field=bool(_EXISTING[row]),target_observation_overlap=bool(_OVERLAP[row]),
        diagnostic_only=False,convergence=dict(converged=passed,probability_delta=delta,quantile_delta=mu_delta,
        joint_delta=r['joint_delta'],log_normalization=r['log_normalization'],marginal_half_delta=r['marginal_half_delta']))
    mass=p.distance_bins().reshape(2,len(_MODEL.weights),3,6);component=mass.sum(axis=(0,2,3))
    record.update(binary_probability=float(mass[1].sum()),component_effective_count=float(1/(component@component)),
        kernel_negative_support_proxy_by_band=(component@norm.cdf(-_MODEL.means[:,:6]/np.sqrt(np.diagonal(_MODEL.covariances[:,:6,:6],axis1=1,axis2=2)))).tolist())
    if _EVIDENCE['parallax_eligible'][row]:record['gaia']=observed_parallax(p.mu,lp,_EVIDENCE['p_corr'][row],_EVIDENCE['gaia_parallax_error'][row])
    record['references']={}
    for source,(mu,error,valid,fraction) in _VIEWS.items():
        if not valid[row]:continue
        broad={1:'specdis',2:'spdist'}.get(int(_EVIDENCE['primary_channel'][row])) if source=='primary' else source
        broad_error=np.hypot(_EVIDENCE[broad+'_sigma_mu'][row],2.) if broad in ('specdis','spdist') else 2.
        record['references'][source]=dict(mu=float(mu[row]),error=float(error[row]),
            logscore=normal_reference_score(p.mu,lp,mu[row],error[row],broad_fraction=fraction[row],broad_error=broad_error))
    arrays.update(draw_probability=r['draw_probability'],draw_ids=_DRAW_IDS,weights=_WEIGHTS,coefficient=r['coefficient'],
        log_evidence_per_draw=r['log_evidence_per_draw'],identity=np.array([identity]),TYPE=np.array(['PSF']))
    atomic_npz(pdf,**arrays)
    result=dict(inputs=_INPUTS,results=[record],pdf_sha256=sha(pdf),source_record_sha256=sha(_SOURCE/f'object_{identity}.json'),
        reference_reproduction_mu_delta=float(abs(base['mu']-reference['mu']).max()),job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,result);return result


def main():
    global _TABLE,_EVIDENCE,_MODEL,_CAL,_THETA,_BETA,_WEIGHTS,_DRAW_IDS,_FIELDS,_EXISTING,_OVERLAP,_VIEWS,_SOURCE,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','diagnostic'],default='pilot')
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();start=time.monotonic();d=ROOT/'data/v3_0'
    _SOURCE=ROOT/'data/v3_1/diagnostics'/a.stage;_OUT=ROOT/'data/v3_1/target_prior'/a.stage;_OUT.mkdir(parents=True,exist_ok=True)
    correction=d/'native_responses/refit_12_s4096';target=d/'shared_targets/refit_12_s4096'
    native=json.loads((correction/'complete.json').read_text());assert native['passed'] and native['source_target_sha256']==sha(target/'complete.json')
    chain=load(correction/'chain_correction.npz');assert sha(correction/'chain_correction.npz')==native['outputs']['chain_correction.npz']
    prepared=d.parent/'baseline_v0/prepared.json';fields=json.loads(prepared.read_text())['fields']
    paths=[_SOURCE/'cohort.json',_SOURCE/'complete.json',correction/'chain_correction.npz',correction/'complete.json',target/'complete.json',
        prepared,d/'cohorts/development.npz',d/'evaluation/development.npz',d/'cohorts/application.npz',d/'uncertainty_refit/draw_12/model.joblib',
        d/'uncertainty/shared_draws.npz',d/'calibration.npz',ROOT/'docs/v3_1_target_prior_protocol.md',ROOT/'scripts/v3_1_target_prior.py']
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['shared_prior.py','radial_products.py','diagnostics.py']]
    paths += [ROOT/f['config'] for f in fields]
    paths += [ROOT/'src/photdis/v3'/n for n in ['inference.py','likelihood.py','target.py','metrics.py','systematics.py','hierarchy.py','fast.py']]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_INPUTS['stage']=a.stage
    marker=_OUT/'inputs.json'
    if marker.exists():assert json.loads(marker.read_text())==_INPUTS
    else:save_json(marker,_INPUTS)
    cohort=json.loads(paths[0].read_text());source_complete=json.loads(paths[1].read_text());assert sha(paths[0])==source_complete['cohort_sha256']
    for name,digest in source_complete['objects'].items():assert sha(_SOURCE/name)==digest
    _TABLE=load(d/'cohorts/development.npz');_EVIDENCE=load(d/'evaluation/development.npz');np.testing.assert_array_equal(_TABLE['identity'],_EVIDENCE['identity'])
    rows=unique_join(np.array(cohort['identities'],'i8'),_TABLE['identity']);assert (rows>=0).all()
    assert (_TABLE['TYPE'][rows]=='PSF').all() and (_EVIDENCE['TYPE'][rows]=='PSF').all()
    _MODEL=inference_model(joblib.load(d/'uncertainty_refit/draw_12/model.joblib'));assert _MODEL.converged
    assert not np.intersect1d(_TABLE['identity'][rows],_MODEL.training_identity).size
    _CAL,_=calibration_vectors(_TABLE,load(d/'uncertainty/shared_draws.npz'),12)
    app=load(d/'cohorts/application.npz');original=np.unique(app['field'])
    bounds={f['index']:json.loads((ROOT/f['config']).read_text())['bounds_deg'] for f in fields if f['index'] in original}
    _FIELDS,_EXISTING=geographic_fields(_TABLE['RA'],_TABLE['DEC'],_TABLE['cell'],bounds)
    _OVERLAP=np.isin(_TABLE['identity'],app['identity'])
    dummy=np.ones((len(original),len(_MODEL.weights),5));hierarchy=FieldPopulation(dummy,dummy,_MODEL,original)
    decoded=[hierarchy.decode(z) for z in chain['samples']];_BETA=np.array([x[1] for x in decoded]);global_logits=np.array([x[3] for x in decoded])
    _THETA={};_WEIGHTS=chain['weights'];_DRAW_IDS=chain['draw_ids']
    for field in np.unique(_FIELDS[rows]):
        if field in original:
            i=int(np.flatnonzero(original==field)[0]);theta=np.array([x[0][i] for x in decoded])
        else:
            rng=np.random.default_rng(np.random.SeedSequence([316101,12,int(field)]))
            local=global_logits+hierarchy.field_sigma*rng.normal(size=global_logits.shape)
            theta=softmax(np.c_[local,np.zeros(len(local))],axis=1).reshape(-1,3,5)
        _THETA[int(field)]=theta
    evaluator._CAL=load(d/'calibration.npz');_VIEWS=evaluator.evidence_views(_EVIDENCE)
    atomic_npz(_OUT/'shared_parameters.npz',draw_ids=_DRAW_IDS,weights=_WEIGHTS,beta=_BETA,
        field=np.array(sorted(_THETA)),theta=np.stack([_THETA[x] for x in sorted(_THETA)],axis=1),
        identity=_TABLE['identity'][rows],TYPE=_TABLE['TYPE'][rows],assigned_field=_FIELDS[rows],existing_field=_EXISTING[rows],target_overlap=_OVERLAP[rows])
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        records=[]
        for result in pool.map(one,rows):
            records.append(result);print('TARGET PRIOR',a.stage,len(records),flush=True)
    values=[r['results'][0] for r in records]
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,objects={p.name:sha(p) for p in sorted(_OUT.glob('object_*.json'))},
        shared_parameters_sha256=sha(_OUT/'shared_parameters.npz'),stars=len(rows),nonconverged=sum(not r['convergence']['converged'] for r in values),
        assigned_old_fields=int(_EXISTING[rows].sum()),new_field_predictions=int((~_EXISTING[rows]).sum()),
        target_observation_overlap=int(_OVERLAP[rows].sum()),job=os.getenv('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-start,
        scope='Seen development diagnostic, conditional response12 shared target prior. All individual evidence normalizations retained; new-cell effects sampled from hypermean. Finite response ESS1 is unresolved.'))


if __name__=='__main__':main()

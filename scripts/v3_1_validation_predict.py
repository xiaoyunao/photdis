#!/usr/bin/env python3
"""Frozen-role, outcome-unopened reference-prior native predictions."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from v3_0_response import observation
from v3_1_luminosity_guard import predict
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.inference import kernel_loglike
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3.systematics import calibration_vectors

_TABLE=_MODELS=_CAL=_PARAMETERS=_OUT=_INPUTS=_AUDIT=None


def class_loglike(model,obs,grid,cal):
    native,_=kernel_loglike(model,obs,grid,calibration_mag=cal)
    binary=np.array([1-model.binary_reference_fraction,model.binary_reference_fraction])
    native=logsumexp(native+np.log(binary)[:,None,None],axis=0)
    conditional=model.weights[:,None]*model.classes
    conditional/=conditional.sum(axis=0)[None]
    return np.array([logsumexp(native+np.log(conditional[:,c])[:,None],axis=0) for c in range(3)])


def one(row):
    identity=int(_TABLE['identity'][row]); marker=_OUT/f'object_{identity}.json'; pdf=marker.with_suffix('.npz')
    if marker.exists():
        record=json.loads(marker.read_text()); assert record['inputs']==_INPUTS and sha(pdf)==record['sha256']
        return record
    assert _TABLE['TYPE'][row]=='PSF'
    obs=observation(_TABLE,row); grid=np.linspace(-20,50,3501)
    curves={name:class_loglike(model,obs,grid,_CAL[name][row]) for name,model in _MODELS.items()}
    variants=dict(nominal=('nominal',np.zeros((1,3,2))),response12=('response12',np.zeros((1,3,2))),
        corrected_point=('response12',_PARAMETERS['point'][None]),corrected_bootstrap=('response12',_PARAMETERS['bootstrap']))
    arrays=dict(identity=np.array([identity]),TYPE=np.array(['PSF']),grid=grid,bootstrap_draw_ids=np.arange(32))
    arrays.update({name+'_class_loglike':value for name,value in curves.items()})
    arrays.update({name+'_class_frequency':model.weights@model.classes for name,model in _MODELS.items()})
    results={}; checks=[]
    fine_grid=np.linspace(-20,50,7001) if row in _AUDIT else None
    fine={name:class_loglike(model,obs,fine_grid,_CAL[name][row]) for name,model in _MODELS.items()} if fine_grid is not None else {}
    if fine: arrays.update(fine_grid=fine_grid,**{name+'_fine_class_loglike':value for name,value in fine.items()})
    for name,(model_name,parameters) in variants.items():
        model=_MODELS[model_name]; frequency=model.weights@model.classes
        value=predict(curves[model_name],grid,frequency,parameters)
        assert abs(logsumexp(value['logpdf']+np.log(trapezoid_weights(grid))))<1e-10
        np.testing.assert_allclose(value['draw_probability'].sum(axis=(1,2)),1.,atol=1e-10,rtol=0)
        arrays.update({name+'_logpdf':value['logpdf'],name+'_draw_probability':value['draw_probability']})
        results[name]=dict(mu=value['mu'].tolist(),class_distance=value['class_distance'].tolist(),
            qnear=float(value['class_distance'][:,:3].sum()))
        if fine:
            refined=predict(fine[model_name],fine_grid,frequency,parameters)
            delta=float(abs(refined['class_distance']-value['class_distance']).max())
            qdelta=float(abs(refined['mu']-value['mu']).max())
            checks.append(dict(identity=identity,variant=name,probability_delta=delta,quantile_delta=qdelta,passed=bool(delta<.003 and qdelta<.03)))
            arrays[name+'_fine_logpdf']=refined['logpdf']
    atomic_npz(pdf,**arrays)
    record=dict(inputs=_INPUTS,identity=identity,TYPE='PSF',role=str(_TABLE['role'][row]),cell=int(_TABLE['cell'][row]),
        numerical_checks=checks,results=results,sha256=sha(pdf),job=os.getenv('SLURM_JOB_ID'),references_read=False)
    save_json(marker,record); return record


def main():
    global _TABLE,_MODELS,_CAL,_PARAMETERS,_OUT,_INPUTS,_AUDIT
    require_compute(); p=argparse.ArgumentParser(); p.add_argument('--cohort',choices=['field','common'],required=True)
    p.add_argument('--workers',type=int,default=32); a=p.parse_args(); assert 1<=a.workers<=64
    root=ROOT/'data/v3_1'; old=ROOT/'data/v3_0'; roles=root/'validation_roles'; fit=root/'luminosity_fit_refined'
    paths=[roles/'complete.json',roles/f'{a.cohort}_photometry.npz',roles/'references.json',
        old/'sed_fit.joblib',old/'uncertainty_refit/draw_12/model.joblib',old/'uncertainty/shared_draws.npz',
        fit/'complete.json',fit/'parameters.npz',root/'luminosity_guard/complete.json',
        ROOT/'docs/v3_1_validation_prediction_protocol.md',ROOT/'scripts/v3_1_validation_predict.py',
        ROOT/'scripts/v3_1_luminosity_guard.py',ROOT/'src/photdis/v3_1/luminosity_log.py',
        ROOT/'src/photdis/v3_1/radial_products.py',ROOT/'src/photdis/v3/inference.py',ROOT/'src/photdis/v3/systematics.py']
    _INPUTS={str(path.relative_to(ROOT)):sha(path) for path in paths}
    _OUT=root/'validation_reference_predictions'/a.cohort; _OUT.mkdir(parents=True,exist_ok=True)
    freeze=_OUT/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text())==_INPUTS
    else: save_json(freeze,_INPUTS)
    rd=json.loads(paths[0].read_text()); assert rd['roles_frozen'] and sha(paths[1])==rd['outputs'][paths[1].name]
    assert sha(paths[2])==rd['outputs']['references.json']
    fd=json.loads(paths[6].read_text()); assert sha(paths[7])==fd['outputs']['parameters.npz']
    guard=json.loads(paths[8].read_text()); assert guard['numerical_passed'] and guard['guard_passed']
    _TABLE=load(paths[1]); assert (_TABLE['TYPE']=='PSF').all()
    assert (_TABLE['field']==-1).all() and len(np.unique(_TABLE['identity']))==len(_TABLE['identity'])
    _MODELS={name:inference_model(joblib.load(path)) for name,path in zip(['nominal','response12'],paths[3:5])}
    for model in _MODELS.values():
        assert model.converged and not np.intersect1d(model.training_identity,_TABLE['identity']).size
    _PARAMETERS=load(paths[7]); assert _PARAMETERS['bootstrap'].shape==(32,3,2)
    _CAL={'nominal':np.zeros_like(_TABLE['flux'])}
    _CAL['response12'],_=calibration_vectors(_TABLE,load(paths[5]),12)
    ranked=sorted(range(len(_TABLE['identity'])),key=lambda i:hashlib.sha256(f"319501:{int(_TABLE['identity'][i])}".encode()).digest())
    _AUDIT=set(ranked[:64]); save_json(_OUT/'audit_identities.json',[int(_TABLE['identity'][i]) for i in ranked[:64]])
    records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        for record in pool.map(one,range(len(_TABLE['identity']))):
            records.append(record)
            if len(records)%128==0: print('VALIDATION PHOTOMETRY',a.cohort,len(records),flush=True)
    checks=[c for r in records for c in r['numerical_checks']]; csv_rows(_OUT/'numerical.csv',checks)
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,objects={f'object_{r["identity"]}.json':sha(_OUT/f'object_{r["identity"]}.json') for r in records},
        outputs={name:sha(_OUT/name) for name in ['audit_identities.json','numerical.csv']},job=os.getenv('SLURM_JOB_ID'),
        cohort=a.cohort,objects_predicted=len(records),all_PSF=True,references_read=False,final_outcomes_opened=False,
        numerical_checks=len(checks),numerical_passed=all(c['passed'] for c in checks),
        scope='Reference-prior photometry-only predictions on frozen roles. Four model identities; full target-prior/method freeze remains pending. Bootstrap correction is shared cut sensitivity, not absolute-scale calibration.'))


if __name__=='__main__':main()

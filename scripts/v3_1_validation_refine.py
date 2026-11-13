#!/usr/bin/env python3
"""All-object numerical sidecar; preserve initial frozen validation predictions."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from v3_0_response import observation
from v3_1_validation_predict import class_loglike
from v3_1_luminosity_guard import predict
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.likelihood import trapezoid_weights

_SOURCE=_DONE=_TABLE=_MODELS=_CAL=_PARAMETERS=_OUT=_INPUTS=None


def one(row):
    identity=int(_TABLE['identity'][row]);name=f'object_{identity}.json';source=_SOURCE/name
    assert sha(source)==_DONE['objects'][name];old=json.loads(source.read_text())
    source_pdf=source.with_suffix('.npz');assert sha(source_pdf)==old['sha256'];saved=load(source_pdf)
    folder=_OUT/f'object_{identity}';folder.mkdir(exist_ok=True);marker=folder/'complete.json'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS
        for name,digest in r['outputs'].items():assert sha(folder/name)==digest
        return r
    previous=old['results'];obs=observation(_TABLE,row);attempts=[]
    variants=dict(nominal=('nominal',np.zeros((1,3,2))),response12=('response12',np.zeros((1,3,2))),
        corrected_point=('response12',_PARAMETERS['point'][None]),corrected_bootstrap=('response12',_PARAMETERS['bootstrap']))
    for step in (.01,.005,.0025,.00125):
        pdf=folder/f'step{step:g}.npz';record=pdf.with_suffix('.json')
        if record.exists():
            result=json.loads(record.read_text());assert result['inputs']==_INPUTS and sha(pdf)==result['sha256']
        else:
            grid=np.linspace(-20,50,round(70/step)+1)
            if step==.01 and 'fine_grid' in saved:
                np.testing.assert_array_equal(grid,saved['fine_grid'])
                curves={n:saved[n+'_fine_class_loglike'] for n in _MODELS}
            else:curves={n:class_loglike(m,obs,grid,_CAL[n][row]) for n,m in _MODELS.items()}
            arrays=dict(identity=np.array([identity]),TYPE=np.array(['PSF']),grid=grid,bootstrap_draw_ids=np.arange(32))
            arrays.update({n+'_class_loglike':v for n,v in curves.items()});results={};checks=[]
            for name,(model_name,parameters) in variants.items():
                model=_MODELS[model_name];value=predict(curves[model_name],grid,model.weights@model.classes,parameters)
                assert abs(logsumexp(value['logpdf']+np.log(trapezoid_weights(grid))))<1e-10
                np.testing.assert_allclose(value['draw_probability'].sum(axis=(1,2)),1.,atol=1e-10,rtol=0)
                arrays.update({name+'_logpdf':value['logpdf'],name+'_draw_probability':value['draw_probability']})
                results[name]=dict(mu=value['mu'].tolist(),class_distance=value['class_distance'].tolist(),qnear=float(value['class_distance'][:,:3].sum()))
                delta=float(abs(value['class_distance']-previous[name]['class_distance']).max())
                qdelta=float(abs(value['mu']-previous[name]['mu']).max())
                checks.append(dict(variant=name,probability_delta=delta,quantile_delta=qdelta,passed=bool(delta<.003 and qdelta<.03)))
            atomic_npz(pdf,**arrays)
            result=dict(inputs=_INPUTS,sha256=sha(pdf),step=step,results=results,checks=checks,passed=all(r['passed'] for r in checks),job=os.getenv('SLURM_JOB_ID'))
            save_json(record,result)
        attempts.append(dict(step=step,marker=record.name,marker_sha256=sha(record),passed=result['passed'],checks=result['checks']))
        print('VALIDATION NUMERICAL',identity,step,result['passed'],flush=True)
        if result['passed']:break
        previous=result['results']
    result=dict(inputs=_INPUTS,outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'},identity=identity,TYPE='PSF',
        role=str(_TABLE['role'][row]),cell=int(_TABLE['cell'][row]),source_record_sha256=sha(source),attempts=attempts,
        selected_step=attempts[-1]['step'],passed=attempts[-1]['passed'],references_read=False,job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,result);return result


def main():
    global _SOURCE,_DONE,_TABLE,_MODELS,_CAL,_PARAMETERS,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['field','common'],required=True);p.add_argument('--workers',type=int,default=32)
    a=p.parse_args();assert 1<=a.workers<=64;root=ROOT/'data/v3_1';old=ROOT/'data/v3_0';_SOURCE=root/'validation_reference_predictions'/a.cohort
    paths=[_SOURCE/'complete.json',root/'validation_roles'/f'{a.cohort}_photometry.npz',old/'sed_fit.joblib',
        old/'uncertainty_refit/draw_12/model.joblib',old/'uncertainty/shared_draws.npz',root/'luminosity_fit_refined/parameters.npz',
        ROOT/'docs/v3_1_validation_numerics.md',ROOT/'scripts/v3_1_validation_refine.py',ROOT/'scripts/v3_1_validation_predict.py',
        ROOT/'scripts/v3_1_luminosity_guard.py',ROOT/'src/photdis/v3_1/luminosity_log.py']
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_DONE=json.loads(paths[0].read_text());assert _DONE['all_PSF'] and not _DONE['references_read']
    for path in paths[1:6]:assert sha(path)==_DONE['inputs'][str(path.relative_to(ROOT))]
    _OUT=root/'validation_reference_refinement'/a.cohort;_OUT.mkdir(parents=True,exist_ok=True);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    _TABLE=load(paths[1]);assert (_TABLE['TYPE']=='PSF').all()
    _MODELS={n:inference_model(joblib.load(p)) for n,p in zip(['nominal','response12'],paths[2:4])}
    _CAL={'nominal':np.zeros_like(_TABLE['flux'])};_CAL['response12'],_=calibration_vectors(_TABLE,load(paths[4]),12)
    _PARAMETERS=load(paths[5]);records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        for r in pool.map(one,range(len(_TABLE['identity']))):records.append(r)
    rows=[dict(identity=r['identity'],TYPE='PSF',role=r['role'],cell=r['cell'],selected_step=r['selected_step'],passed=r['passed'],attempts=len(r['attempts'])) for r in records]
    csv_rows(_OUT/'summary.csv',rows)
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,outputs={'summary.csv':sha(_OUT/'summary.csv')},
        objects={f'object_{r["identity"]}/complete.json':sha(_OUT/f'object_{r["identity"]}/complete.json') for r in records},
        job=os.getenv('SLURM_JOB_ID'),cohort=a.cohort,objects_checked=len(records),passed=all(r['passed'] for r in records),
        unresolved=sum(not r['passed'] for r in records),references_read=False,final_outcomes_opened=False,
        scope='All frozen nativePSF validation objects; separate bounded numerical sidecars, all original flags retained. No physical calibration claim.'))


if __name__=='__main__':main()

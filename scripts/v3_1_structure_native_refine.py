#!/usr/bin/env python3
"""Independent native structure quadrature sidecars; original attempts are immutable."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.stats import norm
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from v3_0_response import observation
from v3_1_validation_predict import class_loglike
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.gaussian_response import class_gaussian_logresponse
from photdis.v3_1.structure import withdrawn_gaussian_logresponse

_TABLE=_MODEL=_CAL=_SOURCE=_DONE=_OUT=_INPUTS=None


def one(row):
    identity=int(_TABLE['identity'][row]);name=f'object_{identity}.json';p=_SOURCE/name
    assert sha(p)==_DONE['objects'][name];old=json.loads(p.read_text());original=p.with_suffix('.npz');assert sha(original)==old['sha256']
    folder=_OUT/f'object_{identity}';folder.mkdir(exist_ok=True);marker=folder/'complete.json'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS and sha(ROOT/r['selected_payload'])==r['selected_sha256'];return r
    source=load(original);previous=source;prefix='fine';attempts=[];selected=original;passed=old['passed']
    if not passed:
        for global_step,local_step in [(.005,.001),(.0025,.0005),(.00125,.00025)]:
            selected=folder/f'step{global_step:g}.npz';record=selected.with_suffix('.json')
            if record.exists():
                rr=json.loads(record.read_text());assert rr['inputs']==_INPUTS and sha(selected)==rr['sha256'];value=load(selected)
            else:
                common=source['common_mu'];grid=np.unique(np.r_[np.linspace(-20,50,round(70/global_step)+1),
                    np.linspace(common.min()-.2,common.max()+.2,int(np.ceil((common.max()-common.min()+.4)/local_step))+1)])
                curve=class_loglike(_MODEL,observation(_TABLE,row),grid,_CAL[row]);lw=np.log(trapezoid_weights(grid))
                reference=curve+norm.logpdf(grid,15,5)[None]+lw[None];z=logsumexp(reference,axis=1);mass=np.exp(reference-z[:,None])
                center=mass@grid;sd=np.sqrt(np.sum(mass*(grid[None]-center[:,None])**2,axis=1));assert (sd>0).all()
                full=class_gaussian_logresponse(curve,grid,source['radial_means'],source['radial_sigmas'],np.zeros((1,3,2)))[0]
                baseline=withdrawn_gaussian_logresponse(z,center,sd,source['radial_means'],source['radial_sigmas'])
                value={k:source[k] for k in ['identity','TYPE','row','radial_means','radial_sigmas','common_mu','class_frequency']}
                for key in ['grid','class_loglike','reference_logevidence','reference_mean','reference_sigma','full_logresponse','gaussian_logresponse']:
                    value['coarse_'+key]=previous[prefix+'_'+key]
                value.update(fine_grid=grid,fine_class_loglike=curve,fine_reference_logevidence=z,fine_reference_mean=center,
                    fine_reference_sigma=sd,fine_full_logresponse=full,fine_gaussian_logresponse=baseline)
                checks=[]
                for branch in ('full','gaussian'):
                    coarse,fine=value['coarse_'+branch+'_logresponse'],value['fine_'+branch+'_logresponse']
                    relevant=np.maximum(coarse,fine)>max(coarse.max(),fine.max())-30;delta=float(abs(coarse-fine)[relevant].max())
                    probabilities=[softmax(v+np.log(source['class_frequency'])[:,None],axis=0) for v in (coarse,fine)]
                    pd=float(abs(probabilities[0]-probabilities[1]).max())
                    checks.append(dict(branch=branch,maximum_relevant_logresponse_delta=delta,maximum_class_probability_delta=pd,passed=bool(delta<.02 and pd<.003)))
                atomic_npz(selected,**value);rr=dict(inputs=_INPUTS,sha256=sha(selected),global_step=global_step,local_step=local_step,
                    checks=checks,passed=all(c['passed'] for c in checks),job=os.getenv('SLURM_JOB_ID'));save_json(record,rr)
            attempts.append(dict(marker=str(record.relative_to(ROOT)),sha256=sha(record),global_step=global_step,passed=rr['passed'],checks=rr['checks']))
            passed=rr['passed'];print('STRUCTURE NATIVE REFINE',identity,global_step,passed,flush=True)
            if passed:break
            previous=value;prefix='fine'
    r=dict(inputs=_INPUTS,identity=identity,row=row,TYPE='PSF',original_passed=old['passed'],passed=passed,attempts=attempts,
        source_marker_sha256=sha(p),source_payload_sha256=sha(original),selected_payload=str(selected.relative_to(ROOT)),selected_sha256=sha(selected),
        outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'},job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,r);return r


def main():
    global _TABLE,_MODEL,_CAL,_SOURCE,_DONE,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--draw',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)));a=p.parse_args();assert 0<=a.draw<32
    root=ROOT/'data/v3_1/structure';old=ROOT/'data/v3_0';_SOURCE=root/f'native/draw_{a.draw:02}'
    paths=[_SOURCE/'complete.json',root/'photometry.npz',old/f'uncertainty_refit/draw_{a.draw:02}/model.joblib',old/'uncertainty/shared_draws.npz',
        ROOT/'scripts/v3_1_structure_native_refine.py',ROOT/'scripts/v3_1_validation_predict.py',ROOT/'docs/v3_1_structure_refinement_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['structure.py','gaussian_response.py','gaussian_entry.py']]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_DONE=json.loads(paths[0].read_text());assert _DONE['all_PSF'] and _DONE['n']==512
    for p in paths[1:4]:assert sha(p)==_DONE['inputs'][str(p.relative_to(ROOT))]
    _TABLE=load(paths[1]);assert (_TABLE['TYPE']=='PSF').all();_MODEL=inference_model(joblib.load(paths[2]));_CAL,_=calibration_vectors(_TABLE,load(paths[3]),a.draw)
    _OUT=root/f'native_refined/draw_{a.draw:02}';_OUT.mkdir(parents=True,exist_ok=True);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(one,range(512)))
    csv_rows(_OUT/'summary.csv',[dict(identity=r['identity'],row=r['row'],TYPE='PSF',original_passed=r['original_passed'],passed=r['passed'],
        attempts=len(r['attempts']),selected_step=r['attempts'][-1]['global_step'] if r['attempts'] else .01) for r in records])
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,objects={f'object_{r["identity"]}/complete.json':sha(_OUT/f'object_{r["identity"]}/complete.json') for r in records},
        outputs={'summary.csv':sha(_OUT/'summary.csv')},response_draw=a.draw,n=512,all_PSF=True,passed=all(r['passed'] for r in records),
        original_failures=sum(not r['original_passed'] for r in records),unresolved=sum(not r['passed'] for r in records),job=os.getenv('SLURM_JOB_ID'),
        scope='Numerical-only sidecar inventory preserving all original structure likelihood/baseline failures. All512rows retained; posterior count/evidence gates still required.'))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""All-kernel target responses with shared correction scales and frozen audits."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import softmax
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from v3_0_response import observation
from v3_1_validation_predict import class_loglike
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3_1.gaussian_response import class_gaussian_logresponse

_TABLE=_MODEL=_CAL=_PARAMETERS=_GRIDS=_PILOT=_PILOT_DONE=_AUDIT=_OUT=_INPUTS=None


def one(row):
    identity=int(_TABLE['identity'][row]);path=_OUT/f'object_{identity}.npz';marker=path.with_suffix('.json')
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS and sha(path)==r['sha256'];return r
    assert _TABLE['TYPE'][row]=='PSF';source=_PILOT/marker.name;arrays={};values=[]
    if marker.name in _PILOT_DONE['objects']:
        assert sha(source)==_PILOT_DONE['objects'][marker.name];record=json.loads(source.read_text())
        assert sha(source.with_suffix('.npz'))==record['sha256'];data=load(source.with_suffix('.npz'))
        np.testing.assert_array_equal(data['identity'],[identity])
        arrays={k:v for k,v in data.items() if k.startswith(('coarse_','fine_'))}
        values=[data['coarse_logresponse'],data['fine_logresponse']]
    else:
        obs=observation(_TABLE,row)
        for name,n in [('coarse',3501)]+([('fine',7001)] if row in _AUDIT else []):
            grid=np.unique(np.r_[_GRIDS[row],np.linspace(-20,50,n)])
            loglike=class_loglike(_MODEL,obs,grid,_CAL[row])
            response=class_gaussian_logresponse(loglike,grid,DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA,_PARAMETERS)
            arrays.update({name+'_grid':grid,name+'_class_loglike':loglike,name+'_logresponse':response});values.append(response)
    scale=float(max(r.max() for r in values));coarse=values[0]
    arrays['coarse_response']=np.exp(coarse-scale)
    delta=probability_delta=None;passed=True
    if len(values)==2:
        fine=values[1];arrays['fine_response']=np.exp(fine-scale)
        relevant=np.maximum(coarse,fine)>scale-30;delta=float(abs(coarse-fine)[relevant].max())
        frequency=_MODEL.weights@_MODEL.classes
        probabilities=[softmax((r+np.log(frequency)[None,:,None]).reshape(len(r),-1),axis=1) for r in values]
        probability_delta=float(abs(probabilities[0]-probabilities[1]).max());passed=delta<.02 and probability_delta<.003
    atomic_npz(path,identity=np.array([identity]),TYPE=np.array(['PSF']),row=np.array([row]),logscale=np.array(scale),
        correction_ids=np.arange(34),**arrays)
    r=dict(inputs=_INPUTS,identity=identity,row=int(row),TYPE='PSF',sha256=sha(path),fine_audited=len(values)==2,
        maximum_relevant_logresponse_delta=delta,maximum_probability_delta=probability_delta,passed=bool(passed),
        reused_pilot=marker.name in _PILOT_DONE['objects'],job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,r);return r


def main():
    global _TABLE,_MODEL,_CAL,_PARAMETERS,_GRIDS,_PILOT,_PILOT_DONE,_AUDIT,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)));a=p.parse_args();assert 0<=a.shard<4
    old=ROOT/'data/v3_0';root=ROOT/'data/v3_1';_PILOT=root/'target_native_pilot';prediction=old/'shared_predictions/application_refit_0/draw_12'
    paths=[_PILOT/'complete.json',_PILOT/'roles.npz',old/'cohorts/application.npz',old/'uncertainty_refit/draw_12/model.joblib',
        prediction/'shared_system.npz',prediction/'complete.json',root/'luminosity_fit_refined/parameters.npz',
        ROOT/'scripts/v3_1_target_native.py',ROOT/'scripts/v3_1_validation_predict.py',ROOT/'docs/v3_1_target_transfer_protocol.md',
        ROOT/'src/photdis/v3_1/gaussian_response.py',ROOT/'src/photdis/v3_1/gaussian_entry.py',ROOT/'src/photdis/v3/inference.py']
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_PILOT_DONE=json.loads(paths[0].read_text());assert _PILOT_DONE['passed']
    assert sha(paths[1])==_PILOT_DONE['outputs']['roles.npz']
    for path in paths[2:7]:assert sha(path)==_PILOT_DONE['inputs'][str(path.relative_to(ROOT))]
    _TABLE=load(paths[2]);roles=load(paths[1]);np.testing.assert_array_equal(roles['identity'],_TABLE['identity'])
    assert len(_TABLE['identity'])==26000 and (_TABLE['TYPE']=='PSF').all() and (roles['TYPE']=='PSF').all()
    _MODEL=inference_model(joblib.load(paths[3]));assert _MODEL.converged
    system=load(paths[4]);np.testing.assert_array_equal(system['identity'],_TABLE['identity']);_CAL=system['calibration_mag']
    parameters=load(paths[6]);_PARAMETERS=np.concatenate([np.zeros((1,3,2)),parameters['point'][None],parameters['bootstrap']])
    rank=sorted(range(26000),key=lambda i:hashlib.sha256(f"319803:{int(_TABLE['identity'][i])}".encode()).digest())
    _AUDIT=set(rank[:128])|set(roles['pilot_rows'].tolist());rows=list(range(a.shard,26000,4));wanted=set(rows);_GRIDS={}
    # Each original chunk is independently authenticated before extracting grids.
    for path in sorted(prediction.glob('chunk_*.npz')):
        if path.stem.endswith('_joint'):continue
        meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['summary_sha256'];summary=load(path)
        joint=path.with_name(path.stem+'_joint.npz');assert sha(joint)==meta['joint_sha256'];packed=load(joint)
        np.testing.assert_array_equal(summary['identity'],_TABLE['identity'][summary['rows']]);assert (summary['TYPE']=='PSF').all()
        _INPUTS[str(path.relative_to(ROOT))]=meta['summary_sha256'];_INPUTS[str(joint.relative_to(ROOT))]=meta['joint_sha256']
        for i,row in enumerate(summary['rows']):
            row=int(row)
            if row not in wanted:continue
            assert row not in _GRIDS;lo,hi=packed['full_grid_offset'][i:i+2];_GRIDS[row]=packed['full_mu'][lo:hi].copy()
    assert set(_GRIDS)==wanted
    _OUT=root/'target_native';_OUT.mkdir(exist_ok=True);folder=_OUT/f'shard_{a.shard}';folder.mkdir(exist_ok=True);freeze=folder/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        for record in pool.map(one,rows):
            records.append(record)
            if len(records)%128==0:print('TARGET NATIVE',a.shard,len(records),flush=True)
    csv_rows(folder/'checks.csv',[{k:r[k] for k in ('identity','row','TYPE','fine_audited','maximum_relevant_logresponse_delta','maximum_probability_delta','passed','reused_pilot')} for r in records])
    save_json(folder/'complete.json',dict(inputs=_INPUTS,objects={f'object_{r["identity"]}.json':sha(_OUT/f'object_{r["identity"]}.json') for r in records},
        outputs={'checks.csv':sha(folder/'checks.csv')},shard=a.shard,shards=4,objects_prepared=len(records),
        fine_audited=sum(r['fine_audited'] for r in records),passed=all(r['passed'] for r in records),all_PSF=True,
        non_PSF_excluded=0,missing_TYPE_excluded=0,job=os.getenv('SLURM_JOB_ID'),references_read=False,
        scope='Native target responses across34corrections, common logscale perobject. All initial numerical failures retained; no target fit.'))


if __name__=='__main__':main()

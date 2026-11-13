#!/usr/bin/env python3
"""Outcome-unopened native response pilot for fixed-kernel target transfer."""
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from scipy.special import softmax
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from v3_0_response import observation
from v3_1_validation_predict import class_loglike
from photdis.stage1 import require_compute
from photdis.baseline_v1 import support
from photdis.v3.controls import optical_magnitudes
from photdis.v3.model import inference_model
from photdis.v3.inference import DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA
from photdis.v3_1.gaussian_response import class_gaussian_logresponse

_TABLE=_MODEL=_CAL=_PARAMETERS=_GRIDS=_OUT=_INPUTS=None


def one(row):
    started=time.monotonic(); identity=int(_TABLE['identity'][row])
    path=_OUT/f'object_{identity}.npz'; marker=path.with_suffix('.json')
    if marker.exists():
        record=json.loads(marker.read_text())
        assert record['inputs']==_INPUTS and sha(path)==record['sha256']
        return record
    assert _TABLE['TYPE'][row]=='PSF'
    obs=observation(_TABLE,row); values=[]; arrays={}
    for name,number in [('coarse',3501),('fine',7001)]:
        grid=np.unique(np.r_[_GRIDS[row],np.linspace(-20,50,number)])
        loglike=class_loglike(_MODEL,obs,grid,_CAL[row])
        response=class_gaussian_logresponse(loglike,grid,DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA,_PARAMETERS)
        arrays.update({name+'_grid':grid,name+'_class_loglike':loglike,name+'_logresponse':response})
        values.append(response)
    coarse,fine=values
    scale=float(max(coarse.max(),fine.max()))
    relevant=np.maximum(coarse,fine)>scale-30
    delta=float(abs(coarse-fine)[relevant].max())
    frequency=_MODEL.weights@_MODEL.classes
    probabilities=[softmax((r+np.log(frequency)[None,:,None]).reshape(len(r),-1),axis=1) for r in values]
    probability_delta=float(abs(probabilities[0]-probabilities[1]).max())
    atomic_npz(path,identity=np.array([identity]),TYPE=np.array(['PSF']),row=np.array([row]),
        logscale=np.array(scale),coarse_response=np.exp(coarse-scale),fine_response=np.exp(fine-scale),
        correction_ids=np.arange(34),**arrays)
    record=dict(inputs=_INPUTS,identity=identity,row=int(row),TYPE='PSF',sha256=sha(path),
        maximum_relevant_logresponse_delta=delta,maximum_probability_delta=probability_delta,
        relevant_cells=int(relevant.sum()),passed=bool(delta<.02 and probability_delta<.003),
        elapsed_seconds=time.monotonic()-started,job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,record); return record


def main():
    global _TABLE,_MODEL,_CAL,_PARAMETERS,_GRIDS,_OUT,_INPUTS
    require_compute(); old=ROOT/'data/v3_0'; root=ROOT/'data/v3_1'
    prediction=old/'shared_predictions/application_refit_0/draw_12'
    paths=[old/'cohorts/application.npz',old/'uncertainty_refit/draw_12/model.joblib',
        prediction/'shared_system.npz',prediction/'complete.json',old/'controls/qrf_grz.joblib',
        root/'luminosity_fit_refined/parameters.npz',root/'luminosity_fit_refined/complete.json',
        root/'candidate_selection_pilot/context_1/complete.json',
        ROOT/'docs/v3_1_target_transfer_protocol.md',ROOT/'scripts/v3_1_target_native_pilot.py',
        ROOT/'scripts/v3_1_validation_predict.py',ROOT/'src/photdis/v3_1/gaussian_response.py',
        ROOT/'src/photdis/v3_1/gaussian_entry.py',ROOT/'src/photdis/v3/inference.py']
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths}
    _TABLE=load(paths[0]); assert len(_TABLE['identity'])==26000 and (_TABLE['TYPE']=='PSF').all()
    _MODEL=inference_model(joblib.load(paths[1])); assert _MODEL.converged
    system=load(paths[2]); np.testing.assert_array_equal(system['identity'],_TABLE['identity']); _CAL=system['calibration_mag']
    parameters=load(paths[5]); done=json.loads(paths[6].read_text()); assert sha(paths[5])==done['outputs']['parameters.npz']
    _PARAMETERS=np.concatenate([np.zeros((1,3,2)),parameters['point'][None],parameters['bootstrap']]); assert _PARAMETERS.shape==(34,3,2)
    rank=sorted(range(26000),key=lambda i:hashlib.sha256(f"319802:{int(_TABLE['identity'][i])}".encode()).digest())
    rows=sorted(set(rank[:128])|set(json.loads(paths[7].read_text())['context_candidate_rows']))
    _GRIDS={}; wanted=set(rows)
    for path in sorted(prediction.glob('chunk_*.npz')):
        if path.stem.endswith('_joint'):continue
        meta=json.loads(path.with_suffix('.json').read_text()); assert sha(path)==meta['summary_sha256']
        summary=load(path); hit=[i for i,r in enumerate(summary['rows']) if int(r) in wanted]
        if not hit:continue
        joint=path.with_name(path.stem+'_joint.npz'); assert sha(joint)==meta['joint_sha256']; packed=load(joint)
        _INPUTS[str(path.relative_to(ROOT))]=meta['summary_sha256']; _INPUTS[str(joint.relative_to(ROOT))]=meta['joint_sha256']
        np.testing.assert_array_equal(_TABLE['identity'][summary['rows']],summary['identity']); assert (summary['TYPE']=='PSF').all()
        for i in hit:
            row=int(summary['rows'][i]); assert row not in _GRIDS
            lo,hi=packed['full_grid_offset'][i:i+2]; _GRIDS[row]=packed['full_mu'][lo:hi].copy()
    assert set(_GRIDS)==wanted
    _OUT=root/'target_native_pilot'; _OUT.mkdir(exist_ok=True); freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    qrf=joblib.load(paths[4]); qrf.nn.n_jobs=1; mag,error=optical_magnitudes(_TABLE)
    rmag=mag[:,1]-2.5*np.log10(_TABLE['transmission'][:,1])
    candidate=(_TABLE['TYPE']=='PSF')&(_TABLE['photometric_system']=='DECam_DR11')&(rmag<20)&(error.max(axis=1)<.1)&support(qrf,mag)
    assert candidate.sum()==3540
    fields=sorted(np.unique(_TABLE['field']).tolist(),key=lambda field:hashlib.sha256(f'319801:{field}'.encode()).digest())[:3]
    atomic_npz(_OUT/'roles.npz',identity=_TABLE['identity'],TYPE=_TABLE['TYPE'],field=_TABLE['field'],
        design_weight=_TABLE['design_weight'],candidate=candidate,heldout=np.isin(_TABLE['field'],fields),pilot_rows=np.array(rows))
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        records=list(pool.map(one,rows))
    csv_rows(_OUT/'checks.csv',[{k:r[k] for k in ('identity','row','TYPE','maximum_relevant_logresponse_delta','maximum_probability_delta','relevant_cells','passed','elapsed_seconds')} for r in records])
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,objects={f'object_{r["identity"]}.json':sha(_OUT/f'object_{r["identity"]}.json') for r in records},
        outputs={name:sha(_OUT/name) for name in ['checks.csv','roles.npz']},objects_checked=len(records),all_PSF=True,
        non_PSF_excluded=0,missing_TYPE_excluded=0,candidate_count=int(candidate.sum()),heldout_fields=fields,
        passed=all(r['passed'] for r in records),job=os.getenv('SLURM_JOB_ID'),references_read=False,
        scope='All-kernel native likelihood, fixed within-class proportions, 34 shared corrections. Numerical pilot only; no target fit or external score.'))
    print('TARGET NATIVE PILOT',len(records),all(r['passed'] for r in records),flush=True)


if __name__=='__main__':main()

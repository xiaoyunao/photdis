#!/usr/bin/env python3
"""Verify a common numerical proposal across all shared correction states."""
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3.selection import OPTICAL
from photdis.v3_1.certified_support import CertifiedColorSupport
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_1.gaussian_importance import shared_gaussian_entry

_TABLE=_MODEL=_CAL=_ROWS=_PARAMETERS=_SUPPORT=_OUT=_INPUTS=_PREVIOUS=None


def one(context):
    started=time.monotonic();row=_ROWS[context];folder=_OUT/f'context_{context}';folder.mkdir(exist_ok=True);marker=folder/'complete.json'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS
        for name,digest in r['outputs'].items():assert sha(folder/name)==digest
        return r
    model=entry_model_with_local_dust(_MODEL,_TABLE['transmission'][row],_TABLE['EBV'][row])
    means=np.r_[DISTANCE_BASIS_MEAN,15.];sigmas=np.r_[DISTANCE_BASIS_SIGMA,5.];attempts=[];previous=None;passed=False
    for draws in (131072,524288,2097152):
        values=[];bounds=[]
        for seed in (319711,319712):
            path=folder/f'q{draws}_seed{seed}.npz';record=path.with_suffix('.json')
            if record.exists():
                r=json.loads(record.read_text());assert r['inputs']==_INPUTS and sha(path)==r['sha256'];data=load(path)
            else:
                begin=time.monotonic();value,audit=shared_gaussian_entry(model,_TABLE['flux_error'][row,OPTICAL],_TABLE['transmission'][row,OPTICAL],
                    _CAL[row,OPTICAL],means,sigmas,_PARAMETERS,support=_SUPPORT,candidate=True,draws=draws,seed=seed)
                data=dict(alpha=value,**audit);atomic_npz(path,**data);save_json(record,dict(inputs=_INPUTS,sha256=sha(path),elapsed_seconds=time.monotonic()-begin))
            values.append(data['alpha']);bounds.append(float(data['pruning_bound'].max()))
        first,second=values;scale=np.maximum(first,.001)
        change=None if previous is None else float(abs(first-previous).max());relative=None if previous is None else float((abs(first-previous)/scale).max())
        independent=float(abs(first-second).max());indrelative=float((abs(first-second)/scale).max())
        passed=change is not None and change<.0005 and independent<.0005 and relative<.01 and indrelative<.01
        attempts.append(dict(draws=draws,previous_delta=change,previous_stabilized_relative=relative,independent_delta=independent,
            independent_stabilized_relative=indrelative,max_pruning_bound=max(bounds),passed=bool(passed)))
        print('SHARED PROPOSAL',context,draws,attempts[-1],flush=True)
        if passed:break
        previous=first
    cross=[];prior_done=json.loads((_PREVIOUS/'complete.json').read_text())
    for name,index in [('reference',0),('point',1),('bootstrap00',2),('bootstrap15',17),('bootstrap31',33)]:
        p=_PREVIOUS/f'context{context}_{name}_dust1/complete.json';assert sha(p)==prior_done['task_markers'][str(p.relative_to(_PREVIOUS))]
        r=json.loads(p.read_text());assert r['passed'];table=p.parent/f'q{r["selected_draws"]}_seed319701.npz';assert sha(table)==r['outputs'][table.name]
        delta=float(abs(first[index]-load(table)['alpha']).max());cross.append(dict(variant=name,max_delta=delta,passed=delta<.001))
    passed=passed and all(r['passed'] for r in cross);csv_rows(folder/'attempts.csv',attempts);csv_rows(folder/'crosschecks.csv',cross)
    result=dict(inputs=_INPUTS,outputs={p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.npz','.csv') or p.name.startswith('q')},
        context=context,identity=int(_TABLE['identity'][row]),TYPE='PSF',selected_draws=attempts[-1]['draws'],attempts=attempts,
        maximum_prior_ratio=float(data['maximum_radial_density_ratio'].max()),crosschecks=cross,passed=bool(passed),
        elapsed_seconds=time.monotonic()-started,job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,result);return result


def main():
    global _TABLE,_MODEL,_CAL,_ROWS,_PARAMETERS,_SUPPORT,_OUT,_INPUTS,_PREVIOUS
    require_compute();root=ROOT/'data/v3_1';old=ROOT/'data/v3_0';_OUT=root/'gaussian_importance_pilot';_OUT.mkdir(exist_ok=True);_PREVIOUS=root/'gaussian_entry_pilot'
    paths=[_PREVIOUS/'complete.json',root/'candidate_selection_pilot/context_1/complete.json',old/'cohorts/application.npz',
        old/'uncertainty_refit/draw_12/model.joblib',old/'uncertainty/shared_draws.npz',old/'controls/qrf_grz.joblib',
        root/'luminosity_fit_refined/parameters.npz',root/'luminosity_fit_refined/complete.json',root/'selection_certificate_audit/complete.json',
        ROOT/'scripts/v3_1_gaussian_importance_pilot.py',ROOT/'docs/v3_1_gaussian_importance_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/name for name in ('gaussian_importance.py','gaussian_entry.py','selection_dust.py','certified_support.py','exact_support.py')]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    assert json.loads(paths[0].read_text())['passed'] and json.loads(paths[8].read_text())['passed']
    _ROWS=json.loads(paths[1].read_text())['context_candidate_rows'];_TABLE=load(paths[2]);assert (_TABLE['TYPE'][_ROWS]=='PSF').all()
    _MODEL=inference_model(joblib.load(paths[3]));assert _MODEL.converged
    _CAL,_=calibration_vectors(_TABLE,load(paths[4]),12);_SUPPORT=CertifiedColorSupport(joblib.load(paths[5]))
    done=json.loads(paths[7].read_text());assert sha(paths[6])==done['outputs']['parameters.npz'];parameters=load(paths[6])
    _PARAMETERS=np.concatenate([np.zeros((1,3,2)),parameters['point'][None],parameters['bootstrap']])
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(one,range(3)))
    csv_rows(_OUT/'summary.csv',[{k:r[k] for k in ('context','identity','TYPE','selected_draws','maximum_prior_ratio','passed','elapsed_seconds')} for r in records])
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,outputs={'summary.csv':sha(_OUT/'summary.csv')},
        contexts={f'context_{i}/complete.json':sha(_OUT/f'context_{i}/complete.json') for i in range(3)},job=os.getenv('SLURM_JOB_ID'),
        passed=all(r['passed'] for r in records),scope='Bounded shared numerical proposal, all34correction states and three original contexts. Scientific Gaussian priors unchanged; no population/outcome score.'))


if __name__=='__main__':main()

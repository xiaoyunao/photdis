#!/usr/bin/env python3
"""Direct selection normalization for every frozen target observation context."""
import argparse
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
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3.selection import OPTICAL
from photdis.v3_1.certified_support import CertifiedColorSupport
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_1.gaussian_importance import shared_gaussian_entry

_TABLE=_MODEL=_CAL=_PARAMETERS=_SUPPORT=_ROLES=_OUT=_INPUTS=None


def envelope(row,candidate):
    identity=int(_TABLE['identity'][row]); name='candidate' if candidate else 'broad'
    folder=_OUT/name/f'object_{identity}'; folder.mkdir(parents=True,exist_ok=True); marker=folder/'complete.json'
    if marker.exists():
        result=json.loads(marker.read_text()); assert result['inputs']==_INPUTS
        for name,digest in result['outputs'].items():assert sha(folder/name)==digest
        return result
    started=time.monotonic(); model=entry_model_with_local_dust(_MODEL,_TABLE['transmission'][row],_TABLE['EBV'][row])
    previous=None; attempts=[]; passed=False
    for draws in (131072,524288,2097152):
        values=[]; audits=[]
        for seed in (319711,319712):
            path=folder/f'q{draws}_seed{seed}.npz'; record=path.with_suffix('.json')
            if record.exists():
                r=json.loads(record.read_text()); assert r['inputs']==_INPUTS and sha(path)==r['sha256']; data=load(path)
            else:
                begin=time.monotonic(); value,audit=shared_gaussian_entry(model,_TABLE['flux_error'][row,OPTICAL],
                    _TABLE['transmission'][row,OPTICAL],_CAL[row,OPTICAL],DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA,
                    _PARAMETERS,support=_SUPPORT if candidate else None,candidate=candidate,draws=draws,seed=seed)
                data=dict(alpha=value,**audit); atomic_npz(path,**data)
                save_json(record,dict(inputs=_INPUTS,sha256=sha(path),elapsed_seconds=time.monotonic()-begin))
            values.append(data['alpha']); audits.append(data)
        first,second=values; scale=np.maximum(first,.001); independent=abs(first-second)
        change=None if previous is None else abs(first-previous)
        physical=bool((first<=1).all() and (second<=1).all())
        passed=change is not None and change.max()<.0005 and independent.max()<.0005 and (change/scale).max()<.01 and (independent/scale).max()<.01 and physical
        attempts.append(dict(draws=draws,previous_delta=None if change is None else float(change.max()),
            previous_stabilized_relative=None if change is None else float((change/scale).max()),
            independent_delta=float(independent.max()),independent_stabilized_relative=float((independent/scale).max()),
            physical_bounds=physical,passed=bool(passed)))
        atomic_npz(folder/f'q{draws}_differences.npz',independent=independent,
            previous=np.full_like(first,np.nan) if change is None else change)
        if passed:break
        previous=first
    csv_rows(folder/'attempts.csv',attempts)
    result=dict(inputs=_INPUTS,outputs={p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.npz','.csv') or p.name.startswith('q')},
        row=int(row),identity=identity,TYPE='PSF',envelope=name,selected_draws=draws,attempts=attempts,
        passed=bool(passed),elapsed_seconds=time.monotonic()-started,job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,result); return result


def one(row):
    assert _TABLE['TYPE'][row]=='PSF'
    results=[envelope(row,False)]
    if _ROLES['candidate'][row]:results.append(envelope(row,True))
    return [dict(row=r['row'],identity=r['identity'],envelope=r['envelope'],passed=r['passed'],selected_draws=r['selected_draws']) for r in results]


def main():
    global _TABLE,_MODEL,_CAL,_PARAMETERS,_SUPPORT,_ROLES,_OUT,_INPUTS
    require_compute(); p=argparse.ArgumentParser(); p.add_argument('--shard',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1))); a=p.parse_args()
    assert 0<=a.shard<16
    root=ROOT/'data/v3_1'; old=ROOT/'data/v3_0'; native=root/'target_native_pilot'
    paths=[root/'gaussian_importance_pilot/complete.json',native/'complete.json',native/'roles.npz',
        old/'cohorts/application.npz',old/'uncertainty_refit/draw_12/model.joblib',
        old/'shared_predictions/application_refit_0/draw_12/shared_system.npz',old/'controls/qrf_grz.joblib',
        root/'luminosity_fit_refined/parameters.npz',root/'luminosity_fit_refined/complete.json',
        ROOT/'scripts/v3_1_target_entry.py',ROOT/'docs/v3_1_target_entry_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/name for name in ('gaussian_importance.py','gaussian_entry.py','selection_dust.py','certified_support.py','exact_support.py')]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert json.loads(paths[0].read_text())['passed']
    native_done=json.loads(paths[1].read_text()); assert sha(paths[2])==native_done['outputs']['roles.npz']
    # Native response numerical failures do not alter photometric admission.
    # They still block target fitting independently of this entry calculation.
    _TABLE=load(paths[3]); _ROLES=load(paths[2]); assert len(_TABLE['identity'])==26000
    np.testing.assert_array_equal(_ROLES['identity'],_TABLE['identity']); assert (_TABLE['TYPE']=='PSF').all() and (_ROLES['TYPE']=='PSF').all()
    assert _ROLES['candidate'].sum()==3540
    _MODEL=inference_model(joblib.load(paths[4])); assert _MODEL.converged
    system=load(paths[5]); np.testing.assert_array_equal(system['identity'],_TABLE['identity']); _CAL=system['calibration_mag']
    _SUPPORT=CertifiedColorSupport(joblib.load(paths[6])); parameters=load(paths[7]); done=json.loads(paths[8].read_text())
    assert sha(paths[7])==done['outputs']['parameters.npz']
    _PARAMETERS=np.concatenate([np.zeros((1,3,2)),parameters['point'][None],parameters['bootstrap']]); assert _PARAMETERS.shape==(34,3,2)
    _OUT=root/'target_entry'; _OUT.mkdir(exist_ok=True)
    # Separate shard input records avoid concurrent creation of a shared freeze.
    folder=_OUT/f'shard_{a.shard:02}'; folder.mkdir(exist_ok=True); freeze=folder/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    rows=list(range(a.shard,26000,16)); records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        for results in pool.map(one,rows):
            records.extend(results)
            if len(records)%128<len(results):print('TARGET ENTRY',a.shard,len(records),flush=True)
    csv_rows(folder/'status.csv',records)
    save_json(folder/'complete.json',dict(inputs=_INPUTS,job=os.getenv('SLURM_JOB_ID'),shard=a.shard,shards=16,
        rows=len(rows),envelopes=len(records),passed=all(r['passed'] for r in records),failed=sum(not r['passed'] for r in records),
        outputs={'status.csv':sha(folder/'status.csv')},all_PSF=True,non_PSF_excluded=0,missing_TYPE_excluded=0,
        scope='Direct observed-context selection integrals only. Numerical failures retained; no target fit or reference score.'))


if __name__=='__main__':main()

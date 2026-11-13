#!/usr/bin/env python3
"""Prepare true-distance joint response sidecars from authenticated native curves."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3_1.gaussian_joint import class_gaussian_joint_logresponse

_SOURCE=_DONE=_PARAMETERS=_OUT=_INPUTS=None


def one(name):
    path=_OUT/name;payload=path.with_suffix('.npz')
    if path.exists():
        record=json.loads(path.read_text());assert record['inputs']==_INPUTS and sha(payload)==record['sha256'];return record
    source=_SOURCE/name;assert sha(source)==_DONE['objects'][name];record=json.loads(source.read_text())
    native_path=source.with_suffix('.npz');assert sha(native_path)==record['sha256'];native=load(native_path)
    assert native['TYPE'][0]=='PSF';arrays={};checks=[]
    for kind in ['coarse']+(['fine'] if record['fine_audited'] else []):
        joint=class_gaussian_joint_logresponse(native[kind+'_class_loglike'],native[kind+'_grid'],
            DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA,_PARAMETERS,DISTANCE_MU_BOUNDARIES)
        delta=float(abs(logsumexp(joint,axis=-1)-native[kind+'_logresponse']).max())
        assert delta<1e-8,'True-distance bins do not conserve the original response'
        arrays[kind+'_joint_response']=np.exp(joint-native['logscale'])
        arrays[kind+'_joint_logresponse']=joint
        checks.append(dict(grid=kind,max_log_total_delta=delta,passed=bool(delta<1e-8)))
    atomic_npz(payload,identity=native['identity'],TYPE=native['TYPE'],row=native['row'],logscale=native['logscale'],
        correction_ids=np.arange(34),distance_boundaries_mu=DISTANCE_MU_BOUNDARIES,**arrays)
    result=dict(inputs=_INPUTS,identity=int(native['identity'][0]),row=int(native['row'][0]),TYPE='PSF',
        source_marker_sha256=sha(source),source_payload_sha256=sha(native_path),sha256=sha(payload),
        fine_audited=record['fine_audited'],checks=checks,passed=all(c['passed'] for c in checks),job=os.getenv('SLURM_JOB_ID'))
    save_json(path,result);return result


def main():
    global _SOURCE,_DONE,_PARAMETERS,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)));a=p.parse_args();assert 0<=a.shard<4
    root=ROOT/'data/v3_1';_SOURCE=root/'target_native'
    paths=[_SOURCE/f'shard_{a.shard}/complete.json',root/'luminosity_fit_refined/parameters.npz',
        ROOT/'scripts/v3_1_target_joint.py',ROOT/'src/photdis/v3_1/gaussian_joint.py',ROOT/'src/photdis/v3_1/gaussian_entry.py',
        ROOT/'docs/v3_1_target_counts_protocol.md']
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_DONE=json.loads(paths[0].read_text());assert _DONE['all_PSF']
    assert sha(paths[1])==_DONE['inputs'][str(paths[1].relative_to(ROOT))]
    parameters=load(paths[1]);_PARAMETERS=np.concatenate([np.zeros((1,3,2)),parameters['point'][None],parameters['bootstrap']])
    _OUT=root/'target_joint';_OUT.mkdir(exist_ok=True);folder=_OUT/f'shard_{a.shard}';folder.mkdir(exist_ok=True);freeze=folder/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        for r in pool.map(one,sorted(_DONE['objects'])):
            records.append(r)
            if len(records)%128==0:print('TRUE DISTANCE JOINT',a.shard,len(records),flush=True)
    csv_rows(folder/'checks.csv',[dict(identity=r['identity'],row=r['row'],TYPE='PSF',fine_audited=r['fine_audited'],
        maximum_log_total_delta=max(c['max_log_total_delta'] for c in r['checks']),passed=r['passed']) for r in records])
    save_json(folder/'complete.json',dict(inputs=_INPUTS,objects={f'object_{r["identity"]}.json':sha(_OUT/f'object_{r["identity"]}.json') for r in records},
        outputs={'checks.csv':sha(folder/'checks.csv')},shard=a.shard,shards=4,objects_prepared=len(records),passed=all(r['passed'] for r in records),
        all_PSF=True,job=os.getenv('SLURM_JOB_ID'),scope='True-distance class/basis/bin response sidecars, not posterior counts. Shared posterior propagation and fine-count acceptance still required.'))


if __name__=='__main__':main()

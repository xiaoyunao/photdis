#!/usr/bin/env python3
"""Full native response and importance correction of compressed target chains."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import joblib
import numpy as np
from scipy.special import logsumexp,softmax
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.native_response import native_response
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.batch_hierarchy import objective_batch
from v3_0_response import observation
D=ROOT/'data/v3_0'
_MODEL=_SOURCE=_CAL=_OUT=_INPUTS=None


def chunk(path):
    path=Path(path);out=_OUT/'chunks'/path.name;marker=out.with_suffix('.json')
    if marker.exists():
        old=json.loads(marker.read_text());assert old['inputs']==_INPUTS and sha(out)==old['sha256'];return str(out)
    meta=json.loads(path.with_suffix('.json').read_text());joint=path.with_name(path.stem+'_joint.npz')
    assert sha(path)==meta['summary_sha256'] and sha(joint)==meta['joint_sha256']
    summary=load(path);packed=load(joint);rows=summary['rows']
    np.testing.assert_array_equal(_SOURCE['identity'][rows],summary['identity']);assert (summary['TYPE']=='PSF').all()
    result=[]
    for i,row in enumerate(rows):
        lo,hi=packed['full_grid_offset'][i:i+2]
        grid=np.unique(np.r_[packed['full_mu'][lo:hi],np.linspace(-20.,50.,1401)])
        result.append(native_response(_MODEL,observation(_SOURCE,row),grid,packed['source_log_evidence'][i],calibration_mag=_CAL[row]))
    values=np.array(result);compact=values.astype('f4')
    relative=float((abs(compact.astype(float)-values)/np.maximum(values,1e-30)).max())
    assert relative<1e-7
    atomic_npz(out,identity=summary['identity'],TYPE=summary['TYPE'],rows=rows,response=compact)
    save_json(marker,dict(inputs=_INPUTS,sha256=sha(out),n=len(rows),float32_relative_error=relative))
    return str(out)


def main():
    global _MODEL,_SOURCE,_CAL,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--draw',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)))
    p.add_argument('--draws',type=int,default=512);p.add_argument('--workers',type=int,default=32)
    a=p.parse_args();start=time.monotonic();nominal=a.draw==-1
    model_path=D/'sed_fit.joblib' if nominal else D/'uncertainty_refit'/f'draw_{a.draw:02}/model.joblib'
    target=D/'targets'/f'application_fit_0_cached_s{a.draws}' if nominal else D/'shared_targets'/f'refit_{a.draw:02}_s{a.draws}'
    prediction=D/'predictions/application_multiband_fit_0' if nominal else D/'shared_predictions/application_refit_0'/f'draw_{a.draw:02}'
    record=json.loads((target/'complete.json').read_text())
    assert record.get('sampler_passed',record['max_rhat']<1.05 and record['minimum_ess']>=100 and not record.get('divergent',0))
    source_path=D/'cohorts/application.npz';_MODEL=inference_model(joblib.load(model_path));_SOURCE=load(source_path)
    old_path=target/'component_response.npz'
    if not old_path.exists():old_path=D/'targets/application_fit_0/component_response.npz'
    old=load(old_path);_CAL=np.zeros((len(_SOURCE['identity']),6))
    paths=[model_path,source_path,target/'complete.json',target/'chains.npz',prediction/'complete.json',old_path,
        ROOT/'scripts/v3_0_native_response.py',ROOT/'src/photdis/v3/native_response.py',ROOT/'src/photdis/v3/batch_hierarchy.py']
    if not nominal:
        system=load(prediction/'shared_system.npz');np.testing.assert_array_equal(system['identity'],_SOURCE['identity'])
        _CAL=system['calibration_mag'];paths.append(prediction/'shared_system.npz')
    _INPUTS={str(x.relative_to(ROOT)):sha(x) for x in paths};_INPUTS.update(vars(a))
    name=f'nominal_s{a.draws}' if nominal else f'refit_{a.draw:02}_s{a.draws}'
    _OUT=D/'native_responses'/name;(_OUT/'chunks').mkdir(parents=True,exist_ok=True)
    if (_OUT/'inputs.json').exists():assert json.loads((_OUT/'inputs.json').read_text())==_INPUTS
    else:save_json(_OUT/'inputs.json',_INPUTS)
    tasks=sorted(x for x in prediction.glob('chunk_*.npz') if not x.stem.endswith('_joint'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        outputs=list(pool.map(chunk,tasks))
    pieces=[load(x) for x in outputs];ids=np.concatenate([x['identity'] for x in pieces]);np.testing.assert_array_equal(ids,old['identity'])
    response=np.concatenate([x['response'] for x in pieces]);rows=old['rows']
    atomic_npz(_OUT/'component_response.npz',response=response,selection=old['selection'],identity=ids,rows=rows,reference_log_evidence=old['reference_log_evidence'])
    chain=load(target/'chains.npz')['samples'];z=chain.transpose(1,0,2).reshape(-1,chain.shape[-1])
    h=FastFieldPopulation(old['response'].sum(axis=3),old['selection'],_MODEL,_SOURCE['field'][rows],_SOURCE['design_weight'][rows])
    old_value=objective_batch(h,z)
    del h
    h=FastFieldPopulation(response.sum(axis=3),old['selection'],_MODEL,_SOURCE['field'][rows],_SOURCE['design_weight'][rows])
    new_value=objective_batch(h,z);difference=old_value-new_value
    weights=softmax(difference);ess=float(1/(weights@weights))
    correction=float(logsumexp(difference)-np.log(len(difference)))
    halves=[float(logsumexp(difference[j::2])-np.log(len(difference[j::2]))) for j in (0,1)]
    atomic_npz(_OUT/'chain_correction.npz',samples=z,draw_ids=np.arange(len(z),dtype='i8'),weights=weights,log_ratio=difference)
    passed=ess>.5*len(z) and abs(halves[0]-halves[1])<.05
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,n=len(rows),passed=bool(passed),importance_ess=ess,posterior_draws=len(z),
        evidence_log_correction=correction,half_log_correction_difference=abs(halves[0]-halves[1]),
        source_target_sha256=sha(target/'complete.json'),outputs={p.name:sha(p) for p in _OUT.glob('*.npz')},
        scope='Unpruned native likelihood and original physical posterior; exact importance correction for numerical response replacement',
        elapsed_seconds=time.monotonic()-start,job=os.getenv('SLURM_JOB_ID')))
    print('NATIVE RESPONSE',a.draw,passed,ess,correction,flush=True)


if __name__=='__main__':main()

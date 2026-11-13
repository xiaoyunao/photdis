#!/usr/bin/env python3
"""Direct broad/narrow entry pilot in three frozen seen-system contexts."""
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
from photdis.v3.selection import OPTICAL
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_1.candidate_selection import candidate_thresholds
from photdis.v3_1.structure import structure_quadrature
from photdis.v3_1.structure_entry import proposal_entry

_TABLE=_MODEL=_CAL=_ROWS=_OUT=_INPUTS=None


def one(context):
    row=_ROWS[context];folder=_OUT/f'context_{context}';folder.mkdir(exist_ok=True)
    model=entry_model_with_local_dust(_MODEL,_TABLE['transmission'][row],_TABLE['EBV'][row])
    common=np.unique(np.concatenate([structure_quadrature(n,64)[0] for n in (16,32,64)]))
    branches={'background':(DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA,15.,6.),
        'structure':(common,np.full(len(common),.02),10+5*np.log10(9.03),.8)}
    results=[]
    for label,(means,sigmas,location,width) in branches.items():
        previous=None;attempts=[]
        for draws in (131072,524288,2097152):
            alphas=[]
            for seed in (320211,320212):
                path=folder/f'{label}_q{draws}_seed{seed}.npz';marker=path.with_suffix('.json')
                if marker.exists():
                    record=json.loads(marker.read_text());assert record['inputs']==_INPUTS and sha(path)==record['sha256'];value=load(path)['alpha']
                else:
                    start=time.monotonic()
                    value,audit=proposal_entry(model,_TABLE['flux_error'][row,OPTICAL],_TABLE['transmission'][row,OPTICAL],
                        _CAL[row,OPTICAL],means,sigmas,proposal_mean=location,proposal_sigma=width,draws=draws,seed=seed)
                    atomic_npz(path,alpha=value,means=means,sigmas=sigmas,**audit)
                    save_json(marker,dict(inputs=_INPUTS,sha256=sha(path),elapsed_seconds=time.monotonic()-start))
                alphas.append(value)
            first,second=alphas;scale=np.maximum(first,.001);ind=abs(first-second)
            delta=None if previous is None else abs(first-previous)
            physical=bool((first<=1).all() and (second<=1).all())
            passed=delta is not None and delta.max()<.0005 and ind.max()<.0005 and (delta/scale).max()<.01 and (ind/scale).max()<.01 and physical
            attempts.append(dict(draws=draws,previous_delta=None if delta is None else float(delta.max()),
                previous_relative=None if delta is None else float((delta/scale).max()),independent_delta=float(ind.max()),
                independent_relative=float((ind/scale).max()),physical=physical,minimum=float(min(first.min(),second.min())),
                maximum=float(max(first.max(),second.max())),passed=bool(passed)))
            print('STRUCTURE ENTRY',context,label,attempts[-1],flush=True)
            if passed:break
            previous=first
        results.append(dict(branch=label,selected_draws=draws,attempts=attempts,passed=bool(passed)))
    csv_rows(folder/'summary.csv',[dict(context=context,row=int(row),identity=int(_TABLE['identity'][row]),TYPE='PSF',
        branch=r['branch'],selected_draws=r['selected_draws'],passed=r['passed'],**{k:v for k,v in r['attempts'][-1].items() if k not in ('draws','passed')}) for r in results])
    record=dict(inputs=_INPUTS,context=context,row=int(row),identity=int(_TABLE['identity'][row]),TYPE='PSF',results=results,
        outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'},passed=all(r['passed'] for r in results),job=os.getenv('SLURM_JOB_ID'))
    save_json(folder/'complete.json',record);return record


def main():
    global _TABLE,_MODEL,_CAL,_ROWS,_OUT,_INPUTS
    require_compute();root=ROOT/'data/v3_1/structure';old=ROOT/'data/v3_0'
    paths=[root/'complete.json',root/'photometry.npz',old/'uncertainty_refit/draw_12/model.joblib',old/'uncertainty/shared_draws.npz',
        ROOT/'scripts/v3_1_structure_entry_pilot.py',ROOT/'docs/v3_1_structure_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/n for n in ('structure.py','structure_entry.py','selection_dust.py','candidate_selection.py')]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};done=json.loads(paths[0].read_text());assert done['all_PSF'] and sha(paths[1])==done['outputs']['photometry.npz']
    _TABLE=load(paths[1]);assert (_TABLE['TYPE']=='PSF').all();_MODEL=inference_model(joblib.load(paths[2]));assert _MODEL.converged
    _CAL,_=calibration_vectors(_TABLE,load(paths[3]),12)
    effective=np.array([candidate_thresholds(_TABLE['flux_error'][i,OPTICAL],_TABLE['transmission'][i,OPTICAL],_CAL[i,OPTICAL])[0] for i in range(len(_TABLE['identity']))])
    rank=np.argsort(np.exp(np.log(effective).mean(axis=1)),kind='stable');_ROWS=rank[[0,len(rank)//2,-1]]
    _OUT=root/'entry_pilot';_OUT.mkdir(exist_ok=True);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(one,range(3)))
    csv_rows(_OUT/'summary.csv',[dict(context=r['context'],row=r['row'],identity=r['identity'],TYPE='PSF',branch=b['branch'],
        selected_draws=b['selected_draws'],passed=b['passed'],**{k:v for k,v in b['attempts'][-1].items() if k not in ('draws','passed')}) for r in records for b in r['results']])
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,outputs={'summary.csv':sha(_OUT/'summary.csv')},
        contexts={f'context_{i}/complete.json':sha(_OUT/f'context_{i}/complete.json') for i in range(3)},
        rows=_ROWS.tolist(),passed=all(r['passed'] for r in records),job=os.getenv('SLURM_JOB_ID'),
        scope='Seen-system observed-context direct entry pilot; no structure score or new validation outcomes.'))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Exact Sobol extension sidecars for failed structure selection integrals."""
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
from photdis.v3.systematics import calibration_vectors
from photdis.v3.selection import OPTICAL
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_1.structure_entry import proposal_entry

_TABLE=_MODEL=_CAL=_SOURCE=_DONE=_OUT=_INPUTS=_ROWS=None


def one(context):
    source=_SOURCE/f'context_{context}/complete.json';assert sha(source)==_DONE['contexts'][str(source.relative_to(_SOURCE))]
    old=json.loads(source.read_text());row=int(_ROWS[context]);identity=int(_TABLE['identity'][row]);assert old['inputs']==_DONE['inputs'] and old['row']==row and old['identity']==identity and old['TYPE']=='PSF'
    folder=_OUT/f'context_{context}';folder.mkdir(exist_ok=True);marker=folder/'complete.json'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS
        for b in r['results']:
            for value in b['selected'].values():assert sha(ROOT/value['path'])==value['sha256']
        return r
    model=entry_model_with_local_dust(_MODEL,_TABLE['transmission'][row],_TABLE['EBV'][row]);results=[]
    for branch in old['results']:
        name=branch['branch'];last=branch['selected_draws'];previous=branch['attempts'][-2]['draws']
        selected={}
        for key,(draws,seed) in zip(('base','second_seed','previous_level'),[(last,320211),(last,320212),(previous,320211)]):
            path=source.parent/f'{name}_q{draws}_seed{seed}.npz';assert sha(path)==old['outputs'][path.name]
            selected[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
        passed=branch['passed'];checks=None
        if not passed:
            first_path=ROOT/selected['base']['path'];data=load(first_path);values=[]
            assert last==2097152
            location=float(data['proposal_mean']);width=float(data['proposal_sigma'])
            new={}
            for seed,key in [(320211,'base'),(320212,'second_seed')]:
                path=folder/f'{name}_q8388608_seed{seed}.npz';record=path.with_suffix('.json')
                if record.exists():
                    r=json.loads(record.read_text());assert r['inputs']==_INPUTS and sha(path)==r['sha256'];value=load(path)['alpha']
                else:
                    start=time.monotonic();value,audit=proposal_entry(model,_TABLE['flux_error'][row,OPTICAL],_TABLE['transmission'][row,OPTICAL],
                        _CAL[row,OPTICAL],data['means'],data['sigmas'],proposal_mean=location,proposal_sigma=width,draws=8388608,seed=seed)
                    atomic_npz(path,alpha=value,means=data['means'],sigmas=data['sigmas'],**audit)
                    save_json(record,dict(inputs=_INPUTS,sha256=sha(path),elapsed_seconds=time.monotonic()-start))
                new[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path));values.append(value)
            first,second=values;scale=np.maximum(first,.001);delta=abs(first-data['alpha']);ind=abs(first-second)
            physical=bool((first<=1).all() and (second<=1).all())
            passed=delta.max()<.0005 and ind.max()<.0005 and (delta/scale).max()<.01 and (ind/scale).max()<.01 and physical
            checks=dict(draws=8388608,previous_delta=float(delta.max()),previous_relative=float((delta/scale).max()),
                independent_delta=float(ind.max()),independent_relative=float((ind/scale).max()),physical=physical,
                minimum=float(min(first.min(),second.min())),maximum=float(max(first.max(),second.max())),passed=bool(passed))
            new['previous_level']=selected['base'];selected=new
            print('STRUCTURE ENTRY REFINE',context,name,checks,flush=True)
        results.append(dict(branch=name,original_passed=branch['passed'],passed=bool(passed),checks=checks,selected=selected))
    r=dict(inputs=_INPUTS,context=context,row=row,identity=identity,TYPE='PSF',source_marker_sha256=sha(source),results=results,
        passed=all(b['passed'] for b in results),original_passed=old['passed'],outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'},job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,r);return r


def main():
    global _TABLE,_MODEL,_CAL,_SOURCE,_DONE,_OUT,_INPUTS,_ROWS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--draw',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',12)));a=p.parse_args()
    assert 0<=a.draw<32 and (not a.pilot or a.draw==12)
    root=ROOT/'data/v3_1/structure';old=ROOT/'data/v3_0';_SOURCE=root/('entry_pilot' if a.pilot else f'entry/draw_{a.draw:02}')
    paths=[_SOURCE/'complete.json',root/'photometry.npz',old/f'uncertainty_refit/draw_{a.draw:02}/model.joblib',old/'uncertainty/shared_draws.npz',
        ROOT/'scripts/v3_1_structure_entry_refine.py',ROOT/'docs/v3_1_structure_entry_refinement_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['structure_entry.py','selection_dust.py','candidate_selection.py']]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_DONE=json.loads(paths[0].read_text())
    for p in paths[1:4]:assert sha(p)==_DONE['inputs'][str(p.relative_to(ROOT))]
    _TABLE=load(paths[1]);assert (_TABLE['TYPE']=='PSF').all();_MODEL=inference_model(joblib.load(paths[2]));_CAL,_=calibration_vectors(_TABLE,load(paths[3]),a.draw)
    _ROWS=np.array(_DONE['rows']) if a.pilot else np.arange(512);assert len(_ROWS)==(3 if a.pilot else 512)
    _OUT=root/('entry_pilot_refined' if a.pilot else f'entry_refined/draw_{a.draw:02}');_OUT.mkdir(parents=True,exist_ok=True);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(8,len(_ROWS)),mp_context=mp.get_context('fork')) as pool:records=list(pool.map(one,range(len(_ROWS))))
    csv_rows(_OUT/'summary.csv',[dict(context=r['context'],row=r['row'],identity=r['identity'],TYPE='PSF',branch=b['branch'],
        original_passed=b['original_passed'],passed=b['passed'],**{k:v for k,v in (b['checks'] or {}).items() if k!='passed'}) for r in records for b in r['results']])
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,contexts={f'context_{i}/complete.json':sha(_OUT/f'context_{i}/complete.json') for i in range(len(_ROWS))},
        outputs={'summary.csv':sha(_OUT/'summary.csv')},response_draw=a.draw,rows=_ROWS.tolist(),n=len(_ROWS),all_PSF=True,
        passed=all(r['passed'] for r in records),original_failures=sum(not r['original_passed'] for r in records),unresolved=sum(not r['passed'] for r in records),
        job=os.getenv('SLURM_JOB_ID'),scope='Same integrand/proposal/seeds; bounded2^23 numerical extension only for originally failing branches. All original files/flags preserved.'))


if __name__=='__main__':main()

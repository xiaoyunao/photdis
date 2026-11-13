#!/usr/bin/env python3
"""Direct512-context entry matrices for every registered shared structure response."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
import v3_1_structure_entry_pilot as core
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--draw',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)));a=p.parse_args();assert 0<=a.draw<32
    root=ROOT/'data/v3_1/structure';old=ROOT/'data/v3_0';model_dir=old/f'uncertainty_refit/draw_{a.draw:02}'
    paths=[root/'complete.json',root/'photometry.npz',root/'entry_pilot_refined/complete.json',model_dir/'complete.json',model_dir/'model.joblib',
        old/'uncertainty/shared_draws.npz',ROOT/'scripts/v3_1_structure_entry.py',ROOT/'scripts/v3_1_structure_entry_pilot.py',ROOT/'docs/v3_1_structure_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['structure.py','structure_entry.py','selection_dust.py','candidate_selection.py']]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};inputs['response_draw']=a.draw
    prep=json.loads(paths[0].read_text());assert prep['all_PSF'] and sha(paths[1])==prep['outputs']['photometry.npz']
    pilot=json.loads(paths[2].read_text());assert pilot['passed'],'Retain pilot failures; no automatic numerical admission'
    md=json.loads(paths[3].read_text());assert md['converged'] and sha(paths[4])==md['sha256']
    core._TABLE=load(paths[1]);assert len(core._TABLE['identity'])==512 and (core._TABLE['TYPE']=='PSF').all()
    core._MODEL=inference_model(joblib.load(paths[4]));core._CAL,_=calibration_vectors(core._TABLE,load(paths[5]),a.draw)
    core._ROWS=np.arange(512);core._INPUTS=inputs;core._OUT=root/f'entry/draw_{a.draw:02}';core._OUT.mkdir(parents=True,exist_ok=True)
    freeze=core._OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(core.one,range(512)))
    csv_rows(core._OUT/'checks.csv',[dict(row=r['row'],identity=r['identity'],TYPE='PSF',branch=b['branch'],selected_draws=b['selected_draws'],
        **{k:v for k,v in b['attempts'][-1].items() if k!='draws'}) for r in records for b in r['results']])
    save_json(core._OUT/'complete.json',dict(inputs=inputs,contexts={f'context_{i}/complete.json':sha(core._OUT/f'context_{i}/complete.json') for i in range(512)},
        outputs={'checks.csv':sha(core._OUT/'checks.csv')},n=512,response_draw=a.draw,passed=all(r['passed'] for r in records),
        failures=sum(not r['passed'] for r in records),all_PSF=True,job=os.getenv('SLURM_JOB_ID'),
        scope='Direct entry for all512observed contexts, all broad/narrow priors. Shared32response/count/evidence/quadrature transfers remain required.'))


if __name__=='__main__':main()

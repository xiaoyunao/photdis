#!/usr/bin/env python3
"""Dust-matched candidate-entry sidecar with bounded numerical refinement."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
import v3_1_candidate_refine as engine
from v2_1_near import ROOT, load, sha, save_json, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.selection import OPTICAL
from photdis.v3_1.exact_support import ExactColorSupport
from photdis.v3_1.selection_dust import entry_model_with_local_dust


def main():
    require_compute(); p=argparse.ArgumentParser()
    p.add_argument('--context',type=int,choices=[0,1,2],required=True)
    p.add_argument('--workers',type=int,default=64); a=p.parse_args(); assert 1<=a.workers<=64
    root=ROOT/'data/v3_1'; old=ROOT/'data/v3_0'
    paths=[root/'candidate_selection_pilot/context_1/complete.json',old/'cohorts/application.npz',
        old/'uncertainty_refit/draw_12/model.joblib',old/'controls/qrf_grz.joblib',old/'uncertainty/shared_draws.npz',
        root/'selection_backend_audit/complete.json',ROOT/'scripts/v3_1_candidate_matched.py',
        ROOT/'scripts/v3_1_candidate_refine.py',ROOT/'src/photdis/v3_1/candidate_quadrature.py',
        ROOT/'src/photdis/v3_1/selection_dust.py',ROOT/'src/photdis/v3_1/exact_support.py',
        ROOT/'docs/v3_1_selection_matched.md']
    inputs={str(path.relative_to(ROOT)):sha(path) for path in paths}
    inputs['context']=a.context
    out=root/'candidate_selection_matched'/f'context_{a.context}'; out.mkdir(parents=True,exist_ok=True)
    marker=out/'complete.json'; freeze=out/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text())==inputs
    else: save_json(freeze,inputs)
    if marker.exists():
        result=json.loads(marker.read_text()); assert result['inputs']==inputs
        for name,digest in result['outputs'].items(): assert sha(out/name)==digest
        return
    original=json.loads(paths[0].read_text()); row=original['context_candidate_rows'][a.context]
    assert json.loads(paths[5].read_text())['passed']
    t=load(paths[1]); assert t['TYPE'][row]=='PSF' and t['photometric_system'][row]=='DECam_DR11'
    cal,_=calibration_vectors(t,load(paths[4]),12)
    context=dict(raw_error=t['flux_error'][row,OPTICAL],transmission=t['transmission'][row,OPTICAL],
        calibration_mag=cal[row,OPTICAL])
    model=inference_model(joblib.load(paths[2])); assert model.converged
    engine._MODEL=entry_model_with_local_dust(model,t['transmission'][row],t['EBV'][row])
    engine._SUPPORT=ExactColorSupport(joblib.load(paths[3])); engine._CONTEXT=context
    engine._OUT=out; engine._INPUTS=inputs
    records=[]; previous=None; passed=False
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        for draws in (512,2048,8192,32768):
            first,bound1=engine.variant(pool,draws,317101,.05)
            second,bound2=engine.variant(pool,draws,317102,.05)
            change=None if previous is None else float(abs(first-previous).max())
            independent=float(abs(second-first).max())
            record=dict(draws=draws,previous_level_delta=change,independent_seed_delta=independent,
                max_pruning_bound=float(max(bound1.max(),bound2.max())),fine_delta=None,passed=False)
            records.append(record)
            if change is not None and change<.003 and independent<.003:
                fine,bound3=engine.variant(pool,draws,317101,.025)
                record['fine_delta']=float(abs(fine-first).max())
                record['passed']=passed=record['fine_delta']<.003
                if passed: break
            previous=first
    csv_rows(out/'comparisons.csv',records)
    save_json(marker,dict(inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.npz','.csv'] or p.name.startswith('q')},
        job=os.getenv('SLURM_JOB_ID'),context=a.context,row=row,identity=int(t['identity'][row]),TYPE='PSF',
        EBV=float(t['EBV'][row]),attempts=records,passed=passed,selected_draws=records[-1]['draws'],
        scope='Native local-dust covariance matched, same hard candidate cuts and response12. Own512 baseline; no comparison across changed covariance is called QMC error. Luminosity correction and whole-cohort interpolation not yet applied.'))


if __name__=='__main__': main()

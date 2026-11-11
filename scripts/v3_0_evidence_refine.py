#!/usr/bin/env python3
"""Refine evidence only, preserving the already sampled physical posterior."""
import argparse
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.structured_evidence import structured_bridge
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--draw',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',0)))
    p.add_argument('--draws',type=int,default=512);p.add_argument('--proposals',type=int,default=8192);a=p.parse_args();start=time.monotonic()
    original=D/'shared_targets'/f'refit_{a.draw:02}_s{a.draws}'
    record=json.loads((original/'complete.json').read_text());assert record['sampler_passed']
    for n,h in record['outputs'].items():assert sha(original/n)==h
    paths=[original/'complete.json',original/'chains.npz',original/'component_response.npz',D/'cohorts/application.npz',
        D/'uncertainty_refit'/f'draw_{a.draw:02}/model.joblib',ROOT/'scripts/v3_0_evidence_refine.py',ROOT/'src/photdis/v3/structured_evidence.py']
    inputs={str(x.relative_to(ROOT)):sha(x) for x in paths};inputs.update(vars(a))
    out=original/f'evidence_structured_{a.proposals}';out.mkdir(exist_ok=True)
    if (out/'inputs.json').exists():assert json.loads((out/'inputs.json').read_text())==inputs
    else:save_json(out/'inputs.json',inputs)
    r=load(paths[2]);source=load(paths[3]);model=inference_model(joblib.load(paths[4]));rows=r['rows']
    np.testing.assert_array_equal(source['identity'][rows],r['identity'])
    h=FastFieldPopulation(r['response'].sum(axis=3),r['selection'],model,source['field'][rows],source['design_weight'][rows])
    chains=load(paths[1])['samples'];samples=chains.transpose(1,0,2).reshape(-1,chains.shape[-1])
    def logdensity(x):return -h.objective(x)[0]-.5*h.dimension*np.log(2*np.pi)
    b=structured_bridge(logdensity,samples,h.k+2,h.k,proposal_draws=a.proposals,seed=314000+a.draw)
    atomic_npz(out/'ratios.npz',posterior=b.pop('log_ratio_posterior'),proposal=b.pop('log_ratio_proposal'))
    b['weighted_reference_constant']=record['bridge']['weighted_reference_constant'];b['log_evidence']=b['log_integral']+b['weighted_reference_constant']
    passed=b['fixed_point_converged'] and b['proposal_bridge_ess']>=100 and b['half_log_integral_difference']<.2
    save_json(out/'complete.json',dict(inputs=inputs,passed=bool(passed),bridge=b,ratio_sha256=sha(out/'ratios.npz'),
        original_target_sha256=sha(original/'complete.json'),chain_sha256=sha(original/'chains.npz'),
        scope='Quadrature proposal refinement only; same posterior chains, prior normalization and data constants',elapsed_seconds=time.monotonic()-start,job=os.getenv('SLURM_JOB_ID')))
    print('STRUCTURED EVIDENCE',a.draw,passed,b,flush=True)


if __name__=='__main__':main()

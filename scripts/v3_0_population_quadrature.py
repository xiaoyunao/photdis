#!/usr/bin/env python3
"""Measured exact-importance alternative when finite HMC marginal errors remain."""
import argparse
import json
import os
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.population_quadrature import integrate_population
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--draw',type=int,required=True)
    p.add_argument('--draws',type=int,default=32768);p.add_argument('--chain-draws',type=int,default=512)
    a=p.parse_args();start=time.monotonic();target=D/'shared_targets'/f'refit_{a.draw:02}_s{a.chain_draws}'
    native=D/'native_responses'/f'refit_{a.draw:02}_s{a.chain_draws}'
    record=json.loads((target/'complete.json').read_text());assert record['sampler_passed']
    response_path=native/'component_response.npz'
    if response_path.exists():assert (native/'complete.json').exists()
    else:response_path=target/'component_response.npz'
    model_path=D/'uncertainty_refit'/f'draw_{a.draw:02}/model.joblib';source_path=D/'cohorts/application.npz'
    paths=[target/'complete.json',target/'chains.npz',response_path,model_path,source_path,
        ROOT/'scripts/v3_0_population_quadrature.py',ROOT/'src/photdis/v3/population_quadrature.py',ROOT/'src/photdis/v3/batch_hierarchy.py',ROOT/'src/photdis/v3/structured_evidence.py']
    inputs={str(x.relative_to(ROOT)):sha(x) for x in paths};inputs.update(vars(a))
    suffix='native' if response_path.parent==native else 'compressed'
    out=D/'population_quadrature'/f'refit_{a.draw:02}_{suffix}_n{a.draws}';out.mkdir(parents=True,exist_ok=True)
    if (out/'inputs.json').exists():assert json.loads((out/'inputs.json').read_text())==inputs
    else:save_json(out/'inputs.json',inputs)
    r=load(response_path);source=load(source_path);rows=r['rows'];model=inference_model(joblib.load(model_path))
    h=FastFieldPopulation(r['response'].sum(axis=3),r['selection'],model,source['field'][rows],source['design_weight'][rows])
    chains=load(target/'chains.npz')['samples'];samples=chains.transpose(1,0,2).reshape(-1,chains.shape[-1])
    with threadpool_limits(limits=8):result=integrate_population(h,samples,draws=a.draws,seed=315001+a.draw)
    atomic_npz(out/'samples.npz',**result)
    record={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    save_json(out/'complete.json',dict(inputs=inputs,**record,outputs={'samples.npz':sha(out/'samples.npz')},
        elapsed_seconds=time.monotonic()-start,job=os.getenv('SLURM_JOB_ID'),
        scope='Exact posterior/proposal weights; this benchmark alone does not establish final per-star marginal precision.'))
    print('POPULATION QUADRATURE',a.draw,record,flush=True)


if __name__=='__main__':main()

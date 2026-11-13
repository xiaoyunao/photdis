#!/usr/bin/env python3
"""One frozen correction/cohort conditional population fit and bridge audit."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from v3_0_parallel_bridge import parallel_bridge
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.sampling import hmc,chain_diagnostics
from photdis.v3_1.fixed_population import FixedKernelFieldPopulation

_HIERARCHY=_FIT=_OUT=_INPUTS=_DRAWS=None


def atomic_joblib(path,payload):
    temporary=path.with_name(path.name+f'.partial_{os.getpid()}')
    joblib.dump(payload,temporary);os.replace(temporary,path)


def chain(index):
    path=_OUT/f'chain_{index}.joblib'
    if path.exists():
        saved=joblib.load(path);assert saved['inputs']==_INPUTS;return saved['record']
    result=hmc(_HIERARCHY.objective,_FIT['map'],_FIT['metric'],draws=_DRAWS,warmup=512,seed=319901+index,target_acceptance=.9)
    atomic_joblib(path,dict(inputs=_INPUTS,record=result));return result


def main():
    global _HIERARCHY,_FIT,_OUT,_INPUTS,_DRAWS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--case',choices=['broad','candidate','holdout'],required=True)
    p.add_argument('--correction',type=int,required=True);p.add_argument('--draws',type=int,choices=[512,2048,8192],default=512)
    a=p.parse_args();assert 0<=a.correction<34;_DRAWS=a.draws;start=time.monotonic()
    root=ROOT/'data/v3_1';folder=root/'target_matrices'
    paths=[folder/'complete.json',folder/'matrices.npz',ROOT/'data/v3_0/uncertainty_refit/draw_12/model.joblib',
        ROOT/'scripts/v3_1_target_fit.py',ROOT/'scripts/v3_0_parallel_bridge.py',ROOT/'docs/v3_1_target_transfer_protocol.md',
        ROOT/'src/photdis/v3_1/fixed_population.py',ROOT/'src/photdis/v3/hierarchy.py',ROOT/'src/photdis/v3/sampling.py',ROOT/'src/photdis/v3/structured_evidence.py']
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_INPUTS.update(case=a.case,correction=a.correction,draws=a.draws,warmup=512,chains=4)
    done=json.loads(paths[0].read_text());assert done['passed'] and done['all_PSF'] and sha(paths[1])==done['outputs']['matrices.npz']
    data=load(paths[1]);model=inference_model(joblib.load(paths[2]));assert model.converged
    mask=np.ones(len(data['identity']),bool)
    if a.case=='candidate':mask=data['candidate']
    elif a.case=='holdout':mask=~data['heldout']
    assert (data['TYPE'][mask]=='PSF').all()
    response=data['response'][mask,a.correction];selection=data['selection_candidate' if a.case=='candidate' else 'selection_broad'][mask,a.correction]
    fields=data['field'][mask];weights=data['design_weight'][mask];weights=weights/weights.mean()
    _HIERARCHY=FixedKernelFieldPopulation(response,selection,model.weights@model.classes,fields,weights)
    evidence_constant=float(weights@data['logscale'][mask]);identity=data['identity'][mask]
    _OUT=root/'target_fits'/a.case/f'correction_{a.correction:02}_s{a.draws}';_OUT.mkdir(parents=True,exist_ok=True)
    freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    marker=_OUT/'complete.json'
    if marker.exists():
        saved=json.loads(marker.read_text());assert saved['inputs']==_INPUTS
        for name,digest in saved['outputs'].items():assert sha(_OUT/name)==digest
        print('CACHED TARGET FIT',a.case,a.correction,a.draws,saved['sampler_passed'],flush=True);return
    map_path=_OUT/'map.joblib'
    if map_path.exists():
        saved=joblib.load(map_path);assert saved['inputs']==_INPUTS;_FIT=saved['fit']
    else:
        _FIT=_HIERARCHY.fit();atomic_joblib(map_path,dict(inputs=_INPUTS,fit=_FIT))
    print('FIXED TARGET MAP',a.case,a.correction,_FIT['objective'],_FIT['gradient_max'],flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(chain,range(4)))
    samples=np.array([r['samples'] for r in records]);diagnostics=chain_diagnostics(samples);divergent=sum(r['divergent'] for r in records)
    sampler_passed=diagnostics['max_rhat']<1.05 and diagnostics['minimum_ess']>=100 and divergent==0
    atomic_npz(_OUT/'chains.npz',samples=samples,rhat=diagnostics['rhat'],ess=diagnostics['ess'],
        acceptance=np.array([r['acceptance'] for r in records]),energy_error=np.array([r['energy_error'] for r in records]))
    chosen=samples.transpose(1,0,2).reshape(-1,samples.shape[-1]);decoded=[_HIERARCHY.decode(z) for z in chosen]
    atomic_npz(_OUT/'population.npz',theta=np.array([r[0] for r in decoded]),global_logits=np.array([r[1] for r in decoded]),
        fields=_HIERARCHY.fields,draw_ids=np.arange(len(chosen)),identity=identity,TYPE=data['TYPE'][mask],
        rows=np.flatnonzero(mask),design_weight=data['design_weight'][mask],normalized_case_weights=weights)
    bridge=None;bridge_passed=False
    if sampler_passed:
        # Include normalized standard-normal latent priors. The common response
        # scale is added after quadrature; it is identical across all34 states.
        bridge=parallel_bridge(lambda z:-_HIERARCHY.objective(z)[0]-.5*_HIERARCHY.dimension*np.log(2*np.pi),
            chosen,_HIERARCHY.k,_HIERARCHY.k,proposal_draws=2048,seed=319911,workers=4)
        atomic_npz(_OUT/'bridge_ratios.npz',log_ratio_posterior=bridge.pop('log_ratio_posterior'),log_ratio_proposal=bridge.pop('log_ratio_proposal'))
        bridge_passed=bridge['fixed_point_converged'] and bridge['proposal_bridge_ess']>=100 and bridge['half_log_integral_difference']<.2
        bridge.update(log_evidence=bridge['log_integral']+evidence_constant,evidence_constant=evidence_constant,passed=bool(bridge_passed))
        save_json(_OUT/'bridge.json',bridge)
    result=dict(inputs=_INPUTS,outputs={p.name:sha(p) for p in _OUT.iterdir() if p.suffix in ('.npz','.joblib') or p.name=='bridge.json'},
        case=a.case,correction=a.correction,correction_kind='zero' if a.correction==0 else 'point' if a.correction==1 else f'bootstrap{a.correction-2:02}',
        n=len(identity),fields=_HIERARCHY.fields.tolist(),dimension=_HIERARCHY.dimension,draws=a.draws,
        sampler_passed=bool(sampler_passed),bridge_passed=bool(bridge_passed),max_rhat=diagnostics['max_rhat'],minimum_ess=diagnostics['minimum_ess'],
        divergent=divergent,case_weight_sum=_HIERARCHY.case_weight_sum,effective_case_count=_HIERARCHY.effective_case_count,
        evidence_constant=evidence_constant,bridge=bridge,job=os.getenv('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-start,
        counts_passed=False,final_product_accepted=False,
        scope='Fixed-kernel conditional class/radial population fit. Generalized design likelihood scaled to sampleN. Sampler/evidence audit only; joint-count and fine-selection sensitivity gates remain mandatory. No external outcomes read.')
    save_json(marker,result);print('FIXED TARGET COMPLETE',a.case,a.correction,a.draws,sampler_passed,bridge_passed,flush=True)


if __name__=='__main__':main()

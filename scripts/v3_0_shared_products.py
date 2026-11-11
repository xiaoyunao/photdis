#!/usr/bin/env python3
"""One response's observed-star marginal CDFs and shared latent count products.

All chunks use the same interleaved chain indices. Counts are merged through
conditional means before covariance, preserving cross-chunk correlations.
"""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.compression import unpack
from photdis.v3.target import joint_coefficients, joint_draw_probabilities
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.population import SharedCounts
from photdis.v3.marginal import marginal_coefficients
from photdis.v3.marginal_cdf import cdf_product
from photdis.v3.marginal_density import native_marginal
from v3_0_response import observation
from photdis.v3.ensemble_counts import merge_chunk_counts
D=ROOT/'data/v3_0'
_MODEL=_SOURCE=_RESPONSE=_THETA=_BETA=_FIELDS=_OUT=_PRED=_INPUTS=_DRAW_IDS=_CALIBRATION=None


def chunk(path):
    path=__import__('pathlib').Path(path); output=_OUT/'chunks'/path.name
    marker=output.with_suffix('.json')
    if marker.exists():
        old=json.loads(marker.read_text());assert old['inputs']==_INPUTS
        for name,digest in old['outputs'].items():assert sha(output.parent/name)==digest
        return str(marker)
    s=load(path); meta=json.loads(path.with_suffix('.json').read_text())
    compact_path=path.with_name(path.stem+'_joint.npz')
    assert sha(path)==meta['summary_sha256'] and sha(compact_path)==meta['joint_sha256']
    packed=load(compact_path)
    order=np.argsort(_RESPONSE['identity']); rows=order[np.searchsorted(_RESPONSE['identity'][order],s['identity'])]
    np.testing.assert_array_equal(_RESPONSE['identity'][rows],s['identity'])
    source_rows=_RESPONSE['rows'][rows];fields=_SOURCE['field'][source_rows]
    counts=SharedCounts(_DRAW_IDS,18,seed=308000+int(s['rows'][0]))
    probability=np.zeros((len(rows),3,6)); half_a=np.zeros_like(probability);half_b=np.zeros_like(probability)
    coefficients=np.zeros((len(rows),len(_MODEL.weights),5,3))
    for field in np.unique(fields):
        fi=int(np.flatnonzero(_FIELDS==field)[0]); use=np.flatnonzero(fields==field)
        coef=joint_coefficients(_MODEL,_THETA[:,fi],_BETA)
        r=_RESPONSE['response'][rows[use]]
        p=joint_draw_probabilities(r,coef)
        probability[use]=counts.add(p.reshape(len(_DRAW_IDS),len(use),18),_DRAW_IDS).reshape(-1,3,6)
        half_a[use]=p[::2].mean(axis=0);half_b[use]=p[1::2].mean(axis=0)
        coefficients[use],_=marginal_coefficients(r,coef)
    records=[];grids=[];cdfs=[];offset=[0];native_grids=[];native_logpdf=[];native_offset=[0]
    for j in range(len(rows)):
        compact=unpack(packed,j)
        result=cdf_product(compact,_MODEL,coefficients[j])
        lo,hi=packed['full_grid_offset'][j:j+2]
        grid=np.unique(np.r_[packed['full_mu'][lo:hi],np.linspace(-20.,50.,1401)])
        lp,lognorm=native_marginal(_MODEL,observation(_SOURCE,source_rows[j]),grid,coefficients[j],
            compact['source_log_evidence'],calibration_mag=_CALIBRATION[source_rows[j]])
        result['native_log_normalization']=lognorm
        native_grids.append(grid);native_logpdf.append(lp);native_offset.append(native_offset[-1]+len(grid))
        grids.append(result.pop('grid'));cdfs.append(result.pop('class_cdf'))
        offset.append(offset[-1]+len(grids[-1]))
        result['discrepancy_probability']=float(result.pop('component_probability')@_MODEL.discrepancy_channel)
        result['joint_integration_delta']=float(abs(result['class_distance']-probability[j]).max())
        records.append(result)
    payload={k:np.asarray([r[k] for r in records]) for k in records[0]}
    payload.update(identity=s['identity'],TYPE=s['TYPE'],ls_id=s['ls_id'],field=fields,
        probability=probability,half_a=half_a,half_b=half_b,supported=s['supported'],
        reference_numerical_converged=s['converged'],band_mask=s['band_mask'],
        full_mu=np.concatenate(native_grids),full_logpdf=np.concatenate(native_logpdf),full_grid_offset=np.asarray(native_offset,'i8'),
        coefficient=coefficients.astype('f4'),cdf_grid=np.concatenate(grids).astype('f4'),
        class_cdf=np.concatenate(cdfs,axis=1).astype('f4'),cdf_offset=np.asarray(offset,'i8'))
    assert (payload['TYPE']=='PSF').all()
    atomic_npz(output,**payload)
    count_path=output.with_name(output.stem+'_counts.npz');atomic_npz(count_path,**counts.result())
    save_json(marker,dict(inputs=_INPUTS,n=len(rows),outputs={output.name:sha(output),count_path.name:sha(count_path)},
        source_joint=str(compact_path.relative_to(ROOT)),source_joint_sha256=sha(compact_path),
        maximum_native_integral_error=float(abs(np.expm1(payload['native_log_normalization'])).max()),
        maximum_integral_error=float(abs(payload['integral']-1).max()),
        maximum_joint_delta=float(payload['joint_integration_delta'].max()),
        maximum_omitted_bound=float(payload['omitted_posterior_bound'].max())))
    return str(marker)


def main():
    global _MODEL,_SOURCE,_RESPONSE,_THETA,_BETA,_FIELDS,_OUT,_PRED,_INPUTS,_DRAW_IDS,_CALIBRATION
    require_compute();p=argparse.ArgumentParser()
    p.add_argument('--draw',type=int,default=-1);p.add_argument('--draws',type=int,default=512)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--limit-chunks',type=int,default=0)
    a=p.parse_args();start=time.monotonic()
    nominal=a.draw==-1
    model_path=D/'sed_fit.joblib' if nominal else D/'uncertainty_refit'/f'draw_{a.draw:02}/model.joblib'
    target=D/'targets'/f'application_fit_0_cached_s{a.draws}' if nominal else D/'shared_targets'/f'refit_{a.draw:02}_s{a.draws}'
    _PRED=D/'predictions/application_multiband_fit_0' if nominal else D/'shared_predictions/application_refit_0'/f'draw_{a.draw:02}'
    record=json.loads((target/'complete.json').read_text())
    assert record['max_rhat']<1.05 and record['minimum_ess']>=100,'Unconverged conditional chains'
    if not nominal:assert record['sampler_passed'],'Divergent response sampler'
    else:assert not record['divergent']
    _MODEL=inference_model(joblib.load(model_path));_SOURCE=load(D/'cohorts/application.npz')
    response_path=target/'component_response.npz'
    if nominal and not response_path.exists():response_path=D/'targets/application_fit_0/component_response.npz'
    _RESPONSE=load(response_path)
    _CALIBRATION=np.zeros((len(_SOURCE['identity']),6))
    if not nominal:
        system=load(_PRED/'shared_system.npz')
        np.testing.assert_array_equal(system['identity'],_SOURCE['identity']);_CALIBRATION=system['calibration_mag']
    np.testing.assert_array_equal(_SOURCE['identity'][_RESPONSE['rows']],_RESPONSE['identity'])
    _FIELDS=np.unique(_SOURCE['field'][_RESPONSE['rows']])
    # Decode has no data dependence beyond field labels; do not duplicate a GB
    # response only to construct the same prior coordinates.
    dummy=np.ones((len(_FIELDS),len(_MODEL.weights),5))
    hierarchy=FieldPopulation(dummy,dummy,_MODEL,_FIELDS)
    chains=load(target/'chains.npz')['samples'];samples=chains.transpose(1,0,2).reshape(-1,chains.shape[-1])
    decoded=[hierarchy.decode(z)[:2] for z in samples]
    _THETA=np.array([x[0] for x in decoded]);_BETA=np.array([x[1] for x in decoded]);_DRAW_IDS=np.arange(len(samples),dtype='i8')
    name=f'nominal_s{a.draws}' if nominal else f'refit_{a.draw:02}_s{a.draws}'
    if a.limit_chunks:name+=f'_pilot{a.limit_chunks}'
    _OUT=D/'shared_products'/name;(_OUT/'chunks').mkdir(parents=True,exist_ok=True)
    paths=[model_path,target/'complete.json',target/'chains.npz',response_path,_PRED/'complete.json',D/'cohorts/application.npz',ROOT/'scripts/v3_0_shared_products.py']
    paths += [ROOT/'src/photdis/v3'/x for x in ('marginal_density.py','marginal_cdf.py','marginal.py','target.py','population.py','ensemble_counts.py')]
    _INPUTS={str(x.relative_to(ROOT)):sha(x) for x in paths};_INPUTS.update(vars(a))
    freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    tasks=sorted(x for x in _PRED.glob('chunk_*.npz') if not x.stem.endswith('_joint'))
    if a.limit_chunks:tasks=tasks[:a.limit_chunks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        markers=[]
        for item in pool.map(chunk,tasks):
            markers.append(item)
            if len(markers)%32==0:print('SHARED PRODUCTS',a.draw,len(markers),len(tasks),flush=True)
    records=[load(_OUT/'chunks'/(__import__('pathlib').Path(x).stem+'_counts.npz')) for x in markers]
    counts=merge_chunk_counts(records);atomic_npz(_OUT/'counts.npz',**counts)
    metadata=[json.loads(__import__('pathlib').Path(x).read_text()) for x in markers]
    errors={k:max(x[k] for x in metadata) for k in ('maximum_native_integral_error','maximum_integral_error','maximum_joint_delta','maximum_omitted_bound')}
    # Per-response MC errors are diagnostics; final response-marginal precision
    # is tested after evidence-weighted combination, not max over conditional runs.
    numerical=all(v<.001 for v in errors.values())
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,n=int(counts['n']),global_response_draw=a.draw,
        conditional_marginal_half_delta=float(counts['marginal_half_delta']),integration_passed=bool(numerical),
        errors=errors,chunks={str(x):sha(x) for x in markers},counts_sha256=sha(_OUT/'counts.npz'),
        target_complete_sha256=sha(target/'complete.json'),elapsed_seconds=time.monotonic()-start,job=os.getenv('SLURM_JOB_ID')))
    print('SHARED PRODUCTS COMPLETE',a.draw,numerical,errors,flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Evidence-weighted finite response posterior with inspectable covariance terms."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax,logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.ensemble_counts import combine_counts,combine_star_means
from photdis.v3.compression import cdf_knots,DISTANCE_MU_BOUNDARIES
D=ROOT/'data/v3_0'
_DIRS=_WEIGHTS=_LOGZ=_OUT=_INPUTS=None


def combine_chunk(name):
    out=_OUT/'chunks'/name;marker=out.with_suffix('.json')
    if marker.exists():
        old=json.loads(marker.read_text());assert old['inputs']==_INPUTS and sha(out)==old['sha256'];return str(marker)
    pieces=[]
    for directory in _DIRS:
        path=directory/'chunks'/name;m=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==m['outputs'][name];pieces.append(load(path))
    for p in pieces:
        np.testing.assert_array_equal(p['identity'],pieces[0]['identity']);assert (p['TYPE']=='PSF').all()
    means=np.array([p['probability'].reshape(-1,18) for p in pieces])
    a=np.array([p['half_a'].reshape(-1,18) for p in pieces]);b=np.array([p['half_b'].reshape(-1,18) for p in pieces])
    mean,mc=combine_star_means(means,a,b,_LOGZ)
    output={k:pieces[0][k] for k in ('identity','TYPE','ls_id','field','supported','band_mask')}
    output['probability']=mean.reshape(-1,3,6);output['qnear']=output['probability'][:,:,:3].sum(axis=(1,2))
    for k in ('binary_probability','discrepancy_probability','omitted_posterior_bound'):
        output[k]=np.einsum('r,ri->i',_WEIGHTS,np.array([p[k] for p in pieces]))
    output['reference_numerical_converged']=np.logical_and.reduce([p['reference_numerical_converged'] for p in pieces])
    offsets=[0];grids=[];cdfs=[];quantiles=[];cdf_errors=[];joint_errors=[]
    native_grids=[];native_logpdf=[];native_offsets=[0]
    for j in range(len(output['identity'])):
        curves=[];native=[]
        for p in pieces:
            a,b=p['full_grid_offset'][j:j+2];native.append((p['full_mu'][a:b],p['full_logpdf'][a:b]))
            lo,hi=p['cdf_offset'][j:j+2];curves.append((p['cdf_grid'][lo:hi],p['class_cdf'][:,lo:hi]))
        # All response products explicitly evaluate the same wide [-20,50] tail.
        assert len({float(x[0]) for x,y in native})==1 and len({float(x[-1]) for x,y in native})==1
        native_grid=np.unique(np.concatenate([x for x,y in native]))
        with np.errstate(divide='ignore'):
            native_lp=logsumexp(np.array([np.interp(native_grid,x,y) for x,y in native])+np.log(_WEIGHTS)[:,None],axis=0)
        native_grids.append(native_grid);native_logpdf.append(native_lp);native_offsets.append(native_offsets[-1]+len(native_grid))
        lower=min(x[0] for x,y in curves);upper=max(x[-1] for x,y in curves)
        grid=np.unique(np.r_[np.linspace(lower,upper,int(np.ceil((upper-lower)/.01))+1),DISTANCE_MU_BOUNDARIES])
        total=np.zeros((3,len(grid)));error=0.
        for weight,(x,y) in zip(_WEIGHTS,curves):
            sampled=np.array([np.interp(grid,x,row,left=0,right=row[-1]) for row in y]);total+=weight*sampled
            error+=weight*max(abs(np.interp(x,grid,row)-old).max() for row,old in zip(sampled,y))
        # Compress the combined class CDF with a directly checked interpolation bound.
        retained={0,len(grid)-1}
        for row in total:
            ix,_=cdf_knots(grid,row,tolerance=1e-4,mandatory=DISTANCE_MU_BOUNDARIES);retained.update(ix)
        ix=np.array(sorted(retained));smallgrid=grid[ix];smallcdf=total[:,ix]
        error+=max(abs(np.interp(grid,smallgrid,row)-old).max() for row,old in zip(smallcdf,total))
        finalcdf=total.sum(axis=0);quantiles.append(np.interp([.025,.16,.5,.84,.975],finalcdf,grid))
        edges=np.r_[grid[0],DISTANCE_MU_BOUNDARIES,grid[-1]]
        mass=np.array([np.diff(np.interp(edges,grid,row)) for row in total])
        joint_errors.append(abs(mass-output['probability'][j]).max());cdf_errors.append(error)
        grids.append(smallgrid);cdfs.append(smallcdf);offsets.append(offsets[-1]+len(smallgrid))
    output.update(full_mu=np.concatenate(native_grids),full_logpdf=np.concatenate(native_logpdf),full_grid_offset=np.asarray(native_offsets,'i8'),
        mu=np.array(quantiles),cdf_grid=np.concatenate(grids).astype('f4'),
        class_cdf=np.concatenate(cdfs,axis=1).astype('f4'),cdf_offset=np.asarray(offsets,'i8'),
        cdf_interpolation_bound=np.array(cdf_errors),joint_integration_delta=np.array(joint_errors))
    atomic_npz(out,**output)
    save_json(marker,dict(inputs=_INPUTS,n=len(output['identity']),sha256=sha(out),final_marginal_half_delta=mc,
        max_cdf_error=float(max(cdf_errors)),max_joint_error=float(max(joint_errors)),
        max_omitted_bound=float(output['omitted_posterior_bound'].max())))
    return str(marker)


def weighted_quantile(values,weights,probabilities):
    order=np.argsort(values);return np.interp(probabilities,np.cumsum(weights[order])/weights.sum(),values[order])


def main():
    global _DIRS,_WEIGHTS,_LOGZ,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--draw-ids',default=','.join(map(str,range(32))))
    p.add_argument('--draws',type=int,default=512);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--name',default='all32_native');p.add_argument('--product-suffix',default='_native')
    a=p.parse_args();start=time.monotonic()
    ids=list(map(int,a.draw_ids.split(',')));assert len(set(ids))==len(ids) and len(ids)>=2
    _DIRS=[D/'shared_products'/(f'refit_{i:02}_s{a.draws}'+a.product_suffix) for i in ids]
    logs=[];inputs={};counts=[];rows=[]
    for draw,directory in zip(ids,_DIRS):
        product=json.loads((directory/'complete.json').read_text())
        target=D/'shared_targets'/f'refit_{draw:02}_s{a.draws}'
        fit=json.loads((target/'complete.json').read_text())
        assert fit['sampler_passed'],'Unverified sampler'
        refined=target/'evidence_structured_8192/complete.json'
        evidence=json.loads(refined.read_text()) if refined.exists() else fit
        assert evidence['passed'],'Unverified bridge evidence'
        if refined.exists():
            assert evidence['original_target_sha256']==sha(target/'complete.json')
            inputs[str(refined.relative_to(ROOT))]=sha(refined)
        correction=0.
        if a.product_suffix.startswith('_native'):
            native=D/'native_responses'/f'refit_{draw:02}_s{a.draws}'/'complete.json'
            nr=json.loads(native.read_text());assert nr['passed'] and nr['source_target_sha256']==sha(target/'complete.json')
            correction=nr['evidence_log_correction'];inputs[str(native.relative_to(ROOT))]=sha(native)
        assert product['integration_passed'],'Unverified native per-star integration'
        assert product['target_complete_sha256']==sha(target/'complete.json')
        assert sha(directory/'counts.npz')==product['counts_sha256']
        counts.append(load(directory/'counts.npz'));logs.append(evidence['bridge']['log_evidence']+correction)
        for path in (directory/'complete.json',target/'complete.json'):inputs[str(path.relative_to(ROOT))]=sha(path)
        rows.append(dict(global_response_draw=draw,log_evidence=logs[-1],
            native_evidence_correction=correction,bridge_half_difference=evidence['bridge']['half_log_integral_difference'],
            bridge_proposal_ess=evidence['bridge']['proposal_bridge_ess']))
    _LOGZ=np.array(logs);_WEIGHTS=softmax(_LOGZ)
    _OUT=D/'ensemble_products'/a.name;(_OUT/'chunks').mkdir(parents=True,exist_ok=True)
    for path in [ROOT/'scripts/v3_0_ensemble.py',ROOT/'src/photdis/v3/ensemble_counts.py']:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    inputs.update(vars(a));_INPUTS=inputs
    if (_OUT/'inputs.json').exists():assert json.loads((_OUT/'inputs.json').read_text())==inputs
    else:save_json(_OUT/'inputs.json',inputs)
    for row,weight in zip(rows,_WEIGHTS):row['posterior_response_weight']=float(weight)
    csv_rows(_OUT/'response_weights.csv',rows)
    combined=combine_counts(counts,_LOGZ)
    # Preserve physical/global draw identity rather than only response list position.
    combined['response_draw_ids']=np.asarray(ids,'i8')
    atomic_npz(_OUT/'counts.npz',**combined)
    tasks=sorted(x.name for x in (_DIRS[0]/'chunks').glob('chunk_*.npz') if not x.stem.endswith('_counts'))
    for directory in _DIRS[1:]:
        assert tasks==sorted(x.name for x in (directory/'chunks').glob('chunk_*.npz') if not x.stem.endswith('_counts'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        markers=list(pool.map(combine_chunk,tasks))
    meta=[json.loads(Path(x).read_text()) for x in markers]
    error={k:max(x[k] for x in meta) for k in ('final_marginal_half_delta','max_cdf_error','max_joint_error','max_omitted_bound')}
    summary=[]
    for c in range(3):
        for b in range(6):
            k=c*6+b;lo,hi=weighted_quantile(combined['count_draws'][:,k],combined['draw_weights'],[.025,.975])
            summary.append(dict(class_index=c,distance_bin=b,mean=float(combined['mean'][k]),lo95=float(lo),hi95=float(hi),
                latent_variance=float(combined['conditional_covariance'][k,k]),target_parameter_variance=float(combined['within_response_covariance'][k,k]),
                response_variance=float(combined['between_response_covariance'][k,k])))
    csv_rows(_OUT/'count_summary.csv',summary)
    passed=error['final_marginal_half_delta']<.005 and max(error[k] for k in error if k!='final_marginal_half_delta')<.001
    save_json(_OUT/'complete.json',dict(inputs=inputs,n=int(combined['n']),passed=bool(passed),errors=error,
        response_effective_sample_size=combined['response_effective_sample_size'],response_draw_ids=ids,
        finite_response_scope='Finite reference bootstrap/calibration prior; low response ESS is unresolved response integration, not high scientific confidence.',
        chunks={str(x):sha(x) for x in markers},counts_sha256=sha(_OUT/'counts.npz'),elapsed_seconds=time.monotonic()-start,job=os.getenv('SLURM_JOB_ID')))
    print('ENSEMBLE COMPLETE',passed,error,'response ESS',combined['response_effective_sample_size'],flush=True)


if __name__=='__main__':main()

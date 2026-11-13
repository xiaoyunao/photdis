#!/usr/bin/env python3
"""Preserved numerical-only development sidecars for every matched response."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from v3_0_response import observation
from v3_1_validation_predict import class_loglike
import v3_0_evaluate as evaluator
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.likelihood import trapezoid_weights,ContinuousPosterior
from photdis.v3.metrics import observed_parallax,normal_reference_score
from photdis.v3_1.radial_products import bin_integrals

_TABLE=_MODEL=_FREQUENCY=_CAL=_EVIDENCE=_VIEWS=_SOURCE=_DONE=_OUT=_INPUTS=None


def evaluate(row,grid):
    native=class_loglike(_MODEL,observation(_TABLE,row),grid,_CAL[row])
    logjoint=native+np.log(_FREQUENCY)[:,None]+norm.logpdf(grid,15.,5.)[None]
    logz=float(logsumexp(logjoint+np.log(trapezoid_weights(grid))[None]));logjoint-=logz
    lp=logsumexp(logjoint,axis=0);probability=bin_integrals(np.exp(logjoint),grid)
    value=dict(log_evidence=logz,mu=ContinuousPosterior(grid,np.exp(lp)[None],0.,{}).quantiles().tolist(),
        class_distance=probability.tolist(),qnear=float(probability[:,:3].sum()),references={})
    if _EVIDENCE['parallax_eligible'][row]:
        value['gaia']=observed_parallax(grid,lp,_EVIDENCE['p_corr'][row],_EVIDENCE['gaia_parallax_error'][row])
    for source,(mu,error,valid,fraction) in _VIEWS.items():
        if not valid[row]:continue
        broad_source={1:'specdis',2:'spdist'}.get(int(_EVIDENCE['primary_channel'][row])) if source=='primary' else source
        broad_error=np.hypot(_EVIDENCE[broad_source+'_sigma_mu'][row],2.) if broad_source in ('specdis','spdist') else 2.
        value['references'][source]=dict(mu=float(mu[row]),error=float(error[row]),
            logscore=normal_reference_score(grid,lp,mu[row],error[row],broad_fraction=fraction[row],broad_error=broad_error))
    return value,lp


def compare(coarse,fine):
    checks=dict(joint_probability_delta=float(abs(np.array(coarse['class_distance'])-fine['class_distance']).max()),
        quantile_delta=float(abs(np.array(coarse['mu'])-fine['mu']).max()),log_evidence_delta=abs(coarse['log_evidence']-fine['log_evidence']),
        reference_logscore_delta=max([abs(coarse['references'][s]['logscore']-r['logscore']) for s,r in fine['references'].items()]+[0.]),
        gaia_logscore_delta=abs(coarse['gaia']['log_score']-fine['gaia']['log_score']) if 'gaia' in fine else 0.,
        gaia_PIT_delta=abs(coarse['gaia']['pit']-fine['gaia']['pit']) if 'gaia' in fine else 0.)
    native=checks['joint_probability_delta']<.003 and checks['quantile_delta']<.03 and checks['log_evidence_delta']<.02
    score=checks['reference_logscore_delta']<.02 and checks['gaia_logscore_delta']<.02 and checks['gaia_PIT_delta']<.003
    return checks,bool(native),bool(score)


def one(row):
    identity=int(_TABLE['identity'][row]);name=f'development/object_{identity}.json';original=_SOURCE/name
    assert sha(original)==_DONE['objects'][name];old=json.loads(original.read_text());payload=original.with_suffix('.npz')
    assert old['inputs']==_DONE['inputs'] and sha(payload)==old['sha256']
    assert old['identity']==identity and old['row']==row and old['TYPE']=='PSF' and old['cohort']=='development'
    folder=_OUT/f'object_{identity}';folder.mkdir(exist_ok=True);marker=folder/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text());assert record['inputs']==_INPUTS
        for path,digest in record['outputs'].items():assert sha(folder/path)==digest
        return record
    selected=old.copy();attempts=[];data=load(payload);grid=data['fine_grid'];lp=data['fine_logpdf'];previous=old['fine']
    if not old['passed']:
        for step in (.005,.0025,.00125):
            path=folder/f'step_{step:g}.npz';record_path=path.with_suffix('.json')
            if record_path.exists():
                selected=json.loads(record_path.read_text());assert selected['inputs']==_INPUTS and sha(path)==selected['sha256']
                finer=load(path);next_grid=finer['fine_grid'];next_lp=finer['fine_logpdf']
            else:
                next_grid=np.unique(np.r_[grid,np.linspace(-20,50,round(70/step)+1)])
                fine,next_lp=evaluate(row,next_grid);checks,native,score=compare(previous,fine)
                atomic_npz(path,identity=np.array([identity]),TYPE=np.array(['PSF']),coarse_grid=grid,coarse_logpdf=lp,fine_grid=next_grid,fine_logpdf=next_lp)
                selected=dict(inputs=_INPUTS,identity=identity,row=row,cohort='development',TYPE='PSF',cell=old['cell'],
                    coarse=previous,fine=fine,checks=checks,native_passed=native,score_passed=score,entry_passed=True,
                    passed=native and score,selection=None,sha256=sha(path),step=step,job=os.getenv('SLURM_JOB_ID'))
                save_json(record_path,selected)
            attempts.append(dict(step=step,passed=selected['passed'],checks=selected['checks']))
            if selected['passed']:break
            previous=selected['fine'];grid=next_grid;lp=next_lp
    result=dict(inputs=_INPUTS,identity=identity,row=row,TYPE='PSF',original_marker=str(original.relative_to(ROOT)),
        original_marker_sha256=sha(original),original_passed=old['passed'],passed=selected['passed'],attempts=attempts,
        selected_record=selected,outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'},job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,result);return result


def main():
    global _TABLE,_MODEL,_FREQUENCY,_CAL,_EVIDENCE,_VIEWS,_SOURCE,_DONE,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)));a=p.parse_args();assert 0<=a.index<64
    root=ROOT/'data/v3_1';old=ROOT/'data/v3_0';base=root/'response_probes';draw=a.index if a.index<32 else a.index-32+1000
    _SOURCE=base/f'draw_{draw:04}';_DONE=json.loads((_SOURCE/'complete.json').read_text());assert _DONE['global_draw_id']==draw and _DONE['all_PSF']
    model_dir=old/f'uncertainty_refit/draw_{draw:02}' if a.index<32 else root/f'response_library/refits/draw_{draw}'
    systems=old/'uncertainty/shared_draws.npz' if a.index<32 else root/'response_library/shared_draws.npz'
    paths=[base/'development_photometry.npz',base/'development_evidence.npz',model_dir/'model.joblib',systems,old/'sed_fit.joblib',old/'calibration.npz']
    for path in paths:assert sha(path)==_DONE['inputs'][str(path.relative_to(ROOT))]
    paths += [_SOURCE/'complete.json',ROOT/'scripts/v3_1_response_probe_refine.py',ROOT/'docs/v3_1_response_probe_refinement_protocol.md',
        ROOT/'scripts/v3_1_validation_predict.py',ROOT/'scripts/v3_0_response.py',ROOT/'scripts/v3_0_evaluate.py',ROOT/'src/photdis/v3/metrics.py',ROOT/'src/photdis/v3_1/radial_products.py']
    _INPUTS={str(path.relative_to(ROOT)):sha(path) for path in paths};_INPUTS.update(index=a.index,global_draw_id=draw)
    _TABLE=load(paths[0]);_EVIDENCE=load(paths[1]);np.testing.assert_array_equal(_TABLE['identity'],_EVIDENCE['identity'])
    assert len(_TABLE['identity'])==256 and (_TABLE['TYPE']=='PSF').all() and (_EVIDENCE['TYPE']=='PSF').all()
    _MODEL=inference_model(joblib.load(paths[2]));nominal=inference_model(joblib.load(paths[4]));_FREQUENCY=nominal.weights@nominal.classes
    _CAL=calibration_vectors(_TABLE,load(systems),draw)[0];evaluator._CAL=load(paths[5]);_VIEWS=evaluator.evidence_views(_EVIDENCE)
    _OUT=base/f'development_refined/draw_{draw:04}';_OUT.mkdir(parents=True,exist_ok=True);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(one,range(256)))
    csv_rows(_OUT/'checks.csv',[dict(identity=r['identity'],TYPE='PSF',original_passed=r['original_passed'],passed=r['passed'],
        selected_step=r['selected_record'].get('step',.01),**r['selected_record']['checks']) for r in records])
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,global_draw_id=draw,index=a.index,all_PSF=True,n=256,
        objects={f'object_{r["identity"]}/complete.json':sha(_OUT/f'object_{r["identity"]}/complete.json') for r in records},
        original_failures=sum(not r['original_passed'] for r in records),unresolved=sum(not r['passed'] for r in records),passed=all(r['passed'] for r in records),
        outputs={'checks.csv':sha(_OUT/'checks.csv')},job=os.getenv('SLURM_JOB_ID')))


if __name__=='__main__':main()

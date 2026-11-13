#!/usr/bin/env python3
"""Matched fixed-prior predictive probes across both finite response libraries."""
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
from photdis.v3.selection import OPTICAL
from photdis.v3_1.radial_products import bin_integrals
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_1.gaussian_importance import shared_gaussian_entry

_TABLES=_MODEL=_FREQUENCY=_CAL=_EVIDENCE=_VIEWS=_GRIDS=_OUT=_INPUTS=None


def one(task):
    cohort,row=task;table=_TABLES[cohort];identity=int(table['identity'][row]);folder=_OUT/cohort;folder.mkdir(exist_ok=True)
    marker=folder/f'object_{identity}.json';payload=marker.with_suffix('.npz')
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS and sha(payload)==r['sha256'];return r
    assert table['TYPE'][row]=='PSF';obs=observation(table,row);values=[];arrays={}
    for kind,number in [('coarse',3501),('fine',7001)]:
        grid=np.unique(np.r_[_GRIDS[cohort,identity],np.linspace(-20,50,number)])
        native=class_loglike(_MODEL,obs,grid,_CAL[cohort][row]);logjoint=native+np.log(_FREQUENCY)[:,None]+norm.logpdf(grid,15.,5.)[None]
        logz=float(logsumexp(logjoint+np.log(trapezoid_weights(grid))[None]));logjoint-=logz;lp=logsumexp(logjoint,axis=0)
        probability=bin_integrals(np.exp(logjoint),grid);q=ContinuousPosterior(grid,np.exp(lp)[None],0.,{}).quantiles()
        value=dict(log_evidence=logz,mu=q.tolist(),class_distance=probability.tolist(),qnear=float(probability[:,:3].sum()),references={})
        if cohort=='development':
            if _EVIDENCE['parallax_eligible'][row]:value['gaia']=observed_parallax(grid,lp,_EVIDENCE['p_corr'][row],_EVIDENCE['gaia_parallax_error'][row])
            for source,(mu,error,valid,fraction) in _VIEWS.items():
                if not valid[row]:continue
                broad_source={1:'specdis',2:'spdist'}.get(int(_EVIDENCE['primary_channel'][row])) if source=='primary' else source
                broad_error=np.hypot(_EVIDENCE[broad_source+'_sigma_mu'][row],2.) if broad_source in ('specdis','spdist') else 2.
                value['references'][source]=dict(mu=float(mu[row]),error=float(error[row]),
                    logscore=normal_reference_score(grid,lp,mu[row],error[row],broad_fraction=fraction[row],broad_error=broad_error))
        values.append(value);arrays.update({kind+'_grid':grid,kind+'_logpdf':lp})
    coarse,fine=values;checks={
        'joint_probability_delta':float(abs(np.array(coarse['class_distance'])-fine['class_distance']).max()),
        'quantile_delta':float(abs(np.array(coarse['mu'])-fine['mu']).max()),
        'log_evidence_delta':abs(coarse['log_evidence']-fine['log_evidence']),
        'reference_logscore_delta':max([abs(coarse['references'][s]['logscore']-r['logscore']) for s,r in fine['references'].items()]+[0.]),
        'gaia_logscore_delta':abs(coarse['gaia']['log_score']-fine['gaia']['log_score']) if 'gaia' in fine else 0.,
        'gaia_PIT_delta':abs(coarse['gaia']['pit']-fine['gaia']['pit']) if 'gaia' in fine else 0.}
    native_passed=checks['joint_probability_delta']<.003 and checks['quantile_delta']<.03 and checks['log_evidence_delta']<.02
    score_passed=checks['reference_logscore_delta']<.02 and checks['gaia_logscore_delta']<.02 and checks['gaia_PIT_delta']<.003
    selection=None;entry_passed=True
    if cohort=='target':
        model=entry_model_with_local_dust(_MODEL,table['transmission'][row],table['EBV'][row]);attempts=[];previous=None
        for draws in (131072,524288,2097152):
            alphas=[]
            for seed in (320011,320012):
                name=f'entry_{draws}_{seed}';alpha,audit=shared_gaussian_entry(model,table['flux_error'][row,OPTICAL],
                    table['transmission'][row,OPTICAL],_CAL[cohort][row,OPTICAL],[15.],[5.],np.zeros((1,3,2)),draws=draws,seed=seed)
                arrays[name]=alpha;alphas.append(alpha)
            first,second=alphas;scale=np.maximum(first,.001);change=None if previous is None else abs(first-previous);independent=abs(first-second)
            entry_passed=change is not None and change.max()<.0005 and independent.max()<.0005 and (change/scale).max()<.01 and (independent/scale).max()<.01 and bool((first<=1).all() and (second<=1).all())
            attempts.append(dict(draws=draws,previous_delta=None if change is None else float(change.max()),independent_delta=float(independent.max()),
                previous_stabilized_relative=None if change is None else float((change/scale).max()),independent_stabilized_relative=float((independent/scale).max()),passed=bool(entry_passed)))
            if entry_passed:break
            previous=first
        alpha=float(_FREQUENCY@first[0,:,0]);assert alpha>0
        selection=dict(attempts=attempts,alpha=alpha,passed=bool(entry_passed),
            conditional_log_evidence=fine['log_evidence']-np.log(alpha))
    atomic_npz(payload,identity=np.array([identity]),TYPE=np.array(['PSF']),**arrays)
    r=dict(inputs=_INPUTS,identity=identity,row=row,cohort=cohort,TYPE='PSF',cell=int(table['cell'][row]),
        sha256=sha(payload),coarse=coarse,fine=fine,checks=checks,selection=selection,
        native_passed=bool(native_passed),score_passed=bool(score_passed),entry_passed=bool(entry_passed),
        passed=bool(native_passed and score_passed and entry_passed),job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,r);return r


def main():
    global _TABLES,_MODEL,_FREQUENCY,_CAL,_EVIDENCE,_VIEWS,_GRIDS,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)));a=p.parse_args();assert 0<=a.index<64
    old=ROOT/'data/v3_0';root=ROOT/'data/v3_1';base=root/'response_probes';new=a.index>=32;draw=a.index-32+1000 if new else a.index
    model_dir=root/f'response_library/refits/draw_{draw}' if new else old/f'uncertainty_refit/draw_{draw:02}'
    systems=root/'response_library/shared_draws.npz' if new else old/'uncertainty/shared_draws.npz'
    paths=[base/'complete.json',base/'target_photometry.npz',base/'development_photometry.npz',base/'development_evidence.npz',
        base/'development_design.json',model_dir/'complete.json',model_dir/'model.joblib',systems,old/'sed_fit.joblib',old/'calibration.npz',
        root/'target_native_pilot/complete.json',root/'diagnostics/diagnostic/complete.json',
        ROOT/'scripts/v3_1_response_probe.py',ROOT/'scripts/v3_1_validation_predict.py',ROOT/'scripts/v3_0_evaluate.py',ROOT/'docs/v3_1_response_probe_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['gaussian_importance.py','gaussian_entry.py','selection_dust.py','radial_products.py']]
    paths += [ROOT/'src/photdis/v3'/n for n in ['inference.py','systematics.py','metrics.py']]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_INPUTS.update(global_draw_id=draw,index=a.index)
    done=json.loads(paths[0].read_text());assert done['all_PSF']
    for path in paths[1:5]:assert sha(path)==done['outputs'][path.name]
    md=json.loads(paths[5].read_text());assert md['converged'] and sha(paths[6])==md['sha256']
    _MODEL=inference_model(joblib.load(paths[6]));assert _MODEL.converged
    nominal=inference_model(joblib.load(paths[8]));_FREQUENCY=nominal.weights@nominal.classes
    _TABLES={'target':load(paths[1]),'development':load(paths[2])};_EVIDENCE=load(paths[3])
    assert all((t['TYPE']=='PSF').all() for t in _TABLES.values()) and (_EVIDENCE['TYPE']=='PSF').all()
    np.testing.assert_array_equal(_EVIDENCE['identity'],_TABLES['development']['identity'])
    assert not np.intersect1d(_MODEL.training_identity,_TABLES['development']['identity']).size
    _CAL={name:calibration_vectors(t,load(systems),draw)[0] for name,t in _TABLES.items()}
    evaluator._CAL=load(paths[9]);_VIEWS=evaluator.evidence_views(_EVIDENCE);_GRIDS={}
    for cohort,done_path,folder in [('target',paths[10],root/'target_native_pilot'),('development',paths[11],root/'diagnostics/diagnostic')]:
        record=json.loads(done_path.read_text())
        for identity in _TABLES[cohort]['identity']:
            name=f'object_{int(identity)}.json';marker=folder/name;assert sha(marker)==record['objects'][name];r=json.loads(marker.read_text());pdf=marker.with_suffix('.npz')
            assert sha(pdf)==r['sha256' if cohort=='target' else 'pdf_sha256'];data=load(pdf)
            _GRIDS[cohort,int(identity)]=data['coarse_grid' if cohort=='target' else 'response12_mu']
    _OUT=base/f'draw_{draw:04}';_OUT.mkdir(exist_ok=True);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    tasks=[(name,i) for name,t in _TABLES.items() for i in range(len(t['identity']))];records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        for r in pool.map(one,tasks):
            records.append(r)
            if len(records)%64==0:print('RESPONSE PROBE',draw,len(records),flush=True)
    target=[r for r in records if r['cohort']=='target'];development=[r for r in records if r['cohort']=='development']
    csv_rows(_OUT/'checks.csv',[dict(cohort=r['cohort'],identity=r['identity'],TYPE='PSF',native_passed=r['native_passed'],score_passed=r['score_passed'],entry_passed=r['entry_passed'],**r['checks']) for r in records])
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,objects={f'{r["cohort"]}/object_{r["identity"]}.json':sha(_OUT/r['cohort']/f'object_{r["identity"]}.json') for r in records},
        outputs={'checks.csv':sha(_OUT/'checks.csv')},index=a.index,global_draw_id=draw,library='new' if new else 'original',all_PSF=True,
        target_rows=len(target),development_rows=len(development),target_passed=all(r['native_passed'] and r['entry_passed'] for r in target),
        development_passed=all(r['native_passed'] and r['score_passed'] for r in development),
        target_conditional_log_evidence=sum(r['selection']['conditional_log_evidence'] for r in target),
        target_training_overlap=int(np.isin(_TABLES['target']['identity'],_MODEL.training_identity).sum()),job=os.getenv('SLURM_JOB_ID'),
        scope='Matched fixed nominal-class/N15sigma5 conditional predictive probes. Not26k population-marginal evidence, new independent validation, or corrected final candidate. All draws and numerical failures retained.'))


if __name__=='__main__':main()

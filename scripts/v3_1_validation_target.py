#!/usr/bin/env python3
"""Frozen final-target photometric predictions; no new reference outcomes."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import logsumexp,softmax
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from v3_0_response import observation
from v3_1_validation_predict import class_loglike
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.likelihood import ContinuousPosterior,trapezoid_weights
from photdis.v3_1.shared_prior import geographic_fields
from photdis.v3_1.target_prediction import marginalize_class_population
from photdis.v3_1.luminosity_log import log_luminosity_transform
from photdis.v3_1.radial_products import bin_integrals

_TABLE=_MODEL=_CAL=_STATES=_THETA=_VIEWS=_ASSIGNED=_EXISTING=_REF=_REFDONE=_OUT=_INPUTS=None


def predict(native,grid,field):
    state_density=[];state_probability=[];state_halves=[];responses=[];coefficients=[];checks=[]
    for index,state in enumerate(_STATES):
        parameters=np.array(state['parameters']);transformed=np.array([log_luminosity_transform(native[c],grid,*parameters[c]) for c in range(3)])
        theta=_THETA[index][int(field)];value=marginalize_class_population(transformed,grid,theta,np.full(len(theta),1/len(theta)))
        state_density.append(value['class_logpdf']);state_probability.append(value['class_distance']);state_halves.append(value['half_class_distance'])
        responses.append(value['log_class_basis_bin_response']);coefficients.append(value['log_coefficient'])
        checks.append(dict(correction=index,joint_delta=value['joint_delta'],log_normalization=value['log_normalization']))
    state_density=np.array(state_density);state_probability=np.array(state_probability);state_halves=np.array(state_halves)
    arrays=dict(state_logresponse=np.array(responses),state_logcoefficient=np.array(coefficients),state_class_distance=state_probability);results={}
    for name,view in _VIEWS.items():
        ids=view['ids'];lw=view['log_weights'];weight=np.exp(lw)
        joint=logsumexp(state_density[ids]+lw[:,None,None],axis=0);normalization=float(logsumexp(joint+np.log(trapezoid_weights(grid))[None]));joint-=normalization
        lp=logsumexp(joint,axis=0);prob=bin_integrals(np.exp(joint),grid);quant=ContinuousPosterior(grid,np.exp(lp)[None],0.,{}).quantiles()
        expected=np.einsum('s,scb->cb',weight,state_probability[ids]);halves=np.einsum('s,shcb->hcb',weight,state_halves[ids]);hd=float(abs(halves[0]-halves[1]).max())
        delta=float(abs(prob-expected).max());passed=abs(normalization)<1e-8 and delta<1e-8 and hd<.005
        arrays.update({name+'_class_logpdf':joint,name+'_logpdf':lp})
        results[name]=dict(mu=quant.tolist(),class_distance=prob.tolist(),qnear=float(prob[:,:3].sum()),
            log_normalization=normalization,joint_delta=delta,marginal_half_delta=hd,internal_passed=bool(passed),inherited_ensemble_accepted=view['accepted'])
    return arrays,results,checks


def one(row):
    identity=int(_TABLE['identity'][row]);folder=_OUT/f'object_{identity}';folder.mkdir(exist_ok=True);marker=folder/'complete.json'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS
        for name,digest in r['outputs'].items():assert sha(folder/name)==digest
        return r
    source=_REF/f'object_{identity}/complete.json';assert sha(source)==_REFDONE['objects'][str(source.relative_to(_REF))]
    ref=json.loads(source.read_text());assert ref['passed'] and ref['TYPE']=='PSF' and ref['identity']==identity
    step=ref['selected_step'];steps=[2*step,step]+[s for s in (.005,.0025,.00125) if s<step];previous=None;attempts=[]
    for step in steps:
        path=folder/f'step{step:g}.npz';record=path.with_suffix('.json')
        if record.exists():
            r=json.loads(record.read_text());assert r['inputs']==_INPUTS and sha(path)==r['sha256']
        else:
            grid=np.linspace(-20,50,round(70/step)+1);existing=source.parent/path.name
            if existing.exists():
                assert sha(existing)==ref['outputs'][existing.name];cached=load(existing);np.testing.assert_array_equal(cached['grid'],grid);native=cached['response12_class_loglike']
            else:native=class_loglike(_MODEL,observation(_TABLE,row),grid,_CAL[row])
            arrays,results,internal=predict(native,grid,int(_ASSIGNED[row]));checks=[]
            for name,value in results.items():
                pd=None if previous is None else float(abs(np.array(value['class_distance'])-previous[name]['class_distance']).max())
                qd=None if previous is None else float(abs(np.array(value['mu'])-previous[name]['mu']).max())
                passed=pd is not None and pd<.003 and qd<.03 and value['internal_passed']
                checks.append(dict(view=name,probability_delta=pd,quantile_delta=qd,marginal_half_delta=value['marginal_half_delta'],passed=bool(passed)))
            atomic_npz(path,identity=np.array([identity]),TYPE=np.array(['PSF']),grid=grid,native_class_loglike=native,**arrays)
            r=dict(inputs=_INPUTS,sha256=sha(path),step=step,results=results,state_checks=internal,checks=checks,
                passed=all(c['passed'] for c in checks),job=os.getenv('SLURM_JOB_ID'))
            save_json(record,r)
        attempts.append(dict(step=step,marker=record.name,marker_sha256=sha(record),passed=r['passed'],checks=r['checks']))
        if r['passed']:break
        previous=r['results']
    result=dict(inputs=_INPUTS,identity=identity,row=row,TYPE='PSF',role=str(_TABLE['role'][row]),assigned_field=int(_ASSIGNED[row]),
        existing_target_field=bool(_EXISTING[row]),attempts=attempts,selected_step=attempts[-1]['step'],numerical_passed=r['passed'],
        inherited_ensemble_accepted=all(v['accepted'] for v in _VIEWS.values()),references_read=False,
        outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'},job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,result);return result


def main():
    global _TABLE,_MODEL,_CAL,_STATES,_THETA,_VIEWS,_ASSIGNED,_EXISTING,_REF,_REFDONE,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['field','common'],required=True);p.add_argument('--case',choices=['candidate','broad'],default='candidate')
    p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=16);a=p.parse_args();assert 1<=a.workers<=32
    root=ROOT/'data/v3_1';old=ROOT/'data/v3_0';ensemble=root/f'target_ensemble/{a.case}';_REF=root/f'validation_reference_refinement/{a.cohort}'
    prepared=ROOT/'data/baseline_v0/prepared.json';fields=json.loads(prepared.read_text())['fields']
    paths=[ensemble/'complete.json',ensemble/'states.json',root/'validation_roles/complete.json',root/f'validation_roles/{a.cohort}_photometry.npz',
        _REF/'complete.json',old/'uncertainty_refit/draw_12/model.joblib',old/'uncertainty/shared_draws.npz',old/'cohorts/application.npz',prepared,
        ROOT/'scripts/v3_1_validation_target.py',ROOT/'scripts/v3_1_validation_predict.py',ROOT/'docs/v3_1_validation_target_protocol.md']
    paths += [ROOT/f['config'] for f in fields]
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['target_prediction.py','luminosity_log.py','shared_prior.py','radial_products.py','gaussian_joint.py']]
    paths += [ROOT/'src/photdis/v3'/n for n in ['inference.py','likelihood.py','systematics.py']]
    _INPUTS={str(path.relative_to(ROOT)):sha(path) for path in paths};_INPUTS.update(case=a.case,cohort=a.cohort)
    done=json.loads(paths[0].read_text());assert sha(paths[1])==done['outputs']['states.json'];_STATES=json.loads(paths[1].read_text())['states']
    assert all(s['passed'] for s in _STATES),'Failed conditional states cannot be silently removed'
    roles=json.loads(paths[2].read_text());assert sha(paths[3])==roles['outputs'][paths[3].name]
    _REFDONE=json.loads(paths[4].read_text());assert _REFDONE['passed'] and not _REFDONE['references_read']
    for path in paths[3:4]+paths[5:7]:assert sha(path)==_REFDONE['inputs'][str(path.relative_to(ROOT))]
    _TABLE=load(paths[3]);_MODEL=inference_model(joblib.load(paths[5]));assert (_TABLE['TYPE']=='PSF').all() and not np.intersect1d(_MODEL.training_identity,_TABLE['identity']).size
    _CAL=calibration_vectors(_TABLE,load(paths[6]),12)[0];app=load(paths[7]);assert not np.intersect1d(app['identity'],_TABLE['identity']).size
    bounds={f['index']:json.loads((ROOT/f['config']).read_text())['bounds_deg'] for f in fields if f['index'] in np.unique(app['field'])}
    _ASSIGNED,geographic_existing=geographic_fields(_TABLE['RA'],_TABLE['DEC'],_TABLE['cell'],bounds);_THETA={};_VIEWS={};fitted_fields=None
    for index,state in enumerate(_STATES):
        pop=ROOT/state['population'];assert sha(pop)==state['population_sha256'];d=load(pop);_THETA[index]={}
        if fitted_fields is None:fitted_fields=d['fields']
        else:np.testing.assert_array_equal(fitted_fields,d['fields'])
        for field in np.unique(_ASSIGNED):
            if field in fitted_fields:theta=d['theta'][:,int(np.flatnonzero(fitted_fields==field)[0])]
            else:
                rng=np.random.default_rng(np.random.SeedSequence([320401,index,int(field)]));local=d['global_logits']+.35*rng.normal(size=d['global_logits'].shape)
                theta=softmax(np.c_[local,np.zeros(len(local))],axis=1).reshape(-1,3,5)
            _THETA[index][int(field)]=theta
    _EXISTING=geographic_existing & np.isin(_ASSIGNED,fitted_fields)
    for view in done['views']:
        name=view['view'];m=ensemble/name/'complete.json';assert sha(m)==done['view_markers'][f'{name}/complete.json'];md=json.loads(m.read_text());vp=m.parent/'base_probabilities.npz'
        assert sha(vp)==md['outputs'][vp.name];v=load(vp);_VIEWS[name]=dict(ids=v['state_ids'],log_weights=v['state_log_weights'],accepted=view['accepted'])
    _OUT=root/f'validation_target/{a.case}/{a.cohort}';_OUT.mkdir(parents=True,exist_ok=True);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    assignment=_OUT/'shared_field_assignment.npz'
    assignment_values=dict(identity=_TABLE['identity'],TYPE=_TABLE['TYPE'],assigned_field=_ASSIGNED,
        existing_target_field=_EXISTING,seed_prefix=np.array([320401]))
    if assignment.exists():
        saved=load(assignment); assert set(saved)==set(assignment_values)
        for key,value in assignment_values.items():np.testing.assert_array_equal(saved[key],value)
    else:atomic_npz(assignment,**assignment_values)
    rows=list(range(len(_TABLE['identity'])))
    if a.pilot:rows=sorted(rows,key=lambda row:hashlib.sha256(f'320411:{int(_TABLE["identity"][row])}'.encode()).digest())[:8]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:records=list(pool.map(one,rows))
    summary='pilot_summary.csv' if a.pilot else 'summary.csv'
    csv_rows(_OUT/summary,[dict(identity=r['identity'],TYPE='PSF',role=r['role'],assigned_field=r['assigned_field'],existing_target_field=r['existing_target_field'],
        selected_step=r['selected_step'],numerical_passed=r['numerical_passed'],inherited_ensemble_accepted=r['inherited_ensemble_accepted']) for r in records])
    save_json(_OUT/('pilot_complete.json' if a.pilot else 'complete.json'),dict(inputs=_INPUTS,case=a.case,cohort=a.cohort,pilot=a.pilot,n=len(records),all_PSF=True,
        objects={f'object_{r["identity"]}/complete.json':sha(_OUT/f'object_{r["identity"]}/complete.json') for r in records},
        outputs={name:sha(_OUT/name) for name in [summary,'shared_field_assignment.npz']},
        numerical_passed=all(r['numerical_passed'] for r in records),inherited_ensemble_accepted=all(v['accepted'] for v in _VIEWS.values()),
        references_read=False,final_outcomes_opened=False,job=os.getenv('SLURM_JOB_ID'),
        scope='Exact named final-target identity, individually normalized population draws and shared newfield offsets. Fixed state weights transferred without newphotometry population update. Pilot is not fullvalidation; no references read.'))


if __name__=='__main__':main()

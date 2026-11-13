#!/usr/bin/env python3
"""Old response12/target identity on frozen new photometry, without reference outcomes."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.special import softmax
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from v3_0_response import observation
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.inference import kernel_loglike
from photdis.v3.systematics import calibration_vectors
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3_1.shared_prior import geographic_fields,marginalize_native
from photdis.v3_1.diagnostics import mode_attribution

_TABLE=_MODEL=_CAL=_THETA=_BETA=_WEIGHTS=_DRAW_IDS=_FIELDS=_EXISTING=_STEPS=_OUT=_INPUTS=None


def one(row):
    identity=int(_TABLE['identity'][row]);folder=_OUT/f'object_{identity}';folder.mkdir(exist_ok=True);marker=folder/'complete.json'
    if marker.exists():
        r=json.loads(marker.read_text());assert r['inputs']==_INPUTS
        for name,digest in r['outputs'].items():assert sha(folder/name)==digest
        return r
    step=_STEPS[row];steps=[2*step,step]+[s for s in (.005,.0025,.00125) if s<step];previous=None;attempts=[];passed=False
    for step in steps:
        path=folder/f'step{step:g}.npz';record=path.with_suffix('.json')
        if record.exists():
            r=json.loads(record.read_text());assert r['inputs']==_INPUTS and sha(path)==r['sha256']
        else:
            grid=np.unique(np.r_[np.linspace(-20,50,round(70/step)+1),DISTANCE_MU_BOUNDARIES])
            likelihood,_=kernel_loglike(_MODEL,observation(_TABLE,row),grid,calibration_mag=_CAL[row])
            value=marginalize_native(_MODEL,grid,likelihood,_THETA[int(_FIELDS[row])],_BETA,_WEIGHTS)
            quantile=value['posterior'].quantiles();probability=value['class_distance']
            pd=None if previous is None else float(abs(probability-previous['class_distance']).max())
            qd=None if previous is None else float(abs(quantile-previous['mu']).max())
            passed=pd is not None and pd<.003 and qd<.03 and value['joint_delta']<1e-8 and abs(value['log_normalization'])<1e-8 and value['marginal_half_delta']<.005
            atomic_npz(path,identity=np.array([identity]),TYPE=np.array(['PSF']),grid=grid,logpdf=value['full_logpdf'],mu=quantile,
                class_distance=probability,draw_probability=value['draw_probability'],draw_ids=_DRAW_IDS,weights=_WEIGHTS,
                coefficient=value['coefficient'],log_evidence_per_draw=value['log_evidence_per_draw'])
            r=dict(inputs=_INPUTS,sha256=sha(path),step=step,mu=quantile.tolist(),class_distance=probability.tolist(),
                qnear=float(probability[:,:3].sum()),probability_delta=pd,quantile_delta=qd,joint_delta=value['joint_delta'],
                log_normalization=value['log_normalization'],marginal_half_delta=value['marginal_half_delta'],passed=bool(passed),
                modes=mode_attribution(value['posterior'],len(_MODEL.weights),getattr(_MODEL,'parent_component',None)),job=os.getenv('SLURM_JOB_ID'))
            save_json(record,r)
        attempts.append(dict(step=step,marker=record.name,marker_sha256=sha(record),passed=r['passed'],
            probability_delta=r['probability_delta'],quantile_delta=r['quantile_delta'],marginal_half_delta=r['marginal_half_delta']))
        passed=r['passed']
        if passed:break
        previous=r
    r=dict(inputs=_INPUTS,outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'},identity=identity,row=row,TYPE='PSF',
        role=str(_TABLE['role'][row]),assigned_field=int(_FIELDS[row]),existing_target_field=bool(_EXISTING[row]),
        attempts=attempts,selected_step=attempts[-1]['step'],passed=passed,references_read=False,job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,r);return r


def main():
    global _TABLE,_MODEL,_CAL,_THETA,_BETA,_WEIGHTS,_DRAW_IDS,_FIELDS,_EXISTING,_STEPS,_OUT,_INPUTS
    require_compute();p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['field','common'],required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    assert 1<=a.workers<=32;root=ROOT/'data/v3_1';old=ROOT/'data/v3_0';native=old/'native_responses/refit_12_s4096';target=old/'shared_targets/refit_12_s4096'
    ref=root/'validation_reference_refinement'/a.cohort;prepared=ROOT/'data/baseline_v0/prepared.json'
    fields=json.loads(prepared.read_text())['fields']
    paths=[root/'validation_roles/complete.json',root/f'validation_roles/{a.cohort}_photometry.npz',ref/'complete.json',
        native/'complete.json',native/'chain_correction.npz',target/'complete.json',old/'uncertainty_refit/draw_12/model.joblib',
        old/'uncertainty/shared_draws.npz',old/'cohorts/application.npz',prepared,
        ROOT/'scripts/v3_1_validation_old_target.py',ROOT/'docs/v3_1_validation_old_target_protocol.md']
    paths += [ROOT/f['config'] for f in fields]
    paths += [ROOT/'src/photdis/v3_1'/n for n in ['shared_prior.py','diagnostics.py','radial_products.py']]
    paths += [ROOT/'src/photdis/v3'/n for n in ['inference.py','hierarchy.py','systematics.py','target.py','likelihood.py']]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};_INPUTS['cohort']=a.cohort
    roles=json.loads(paths[0].read_text());assert sha(paths[1])==roles['outputs'][paths[1].name]
    rd=json.loads(paths[2].read_text());assert rd['passed'] and not rd['references_read'] and sha(paths[1])==rd['inputs'][str(paths[1].relative_to(ROOT))]
    nd=json.loads(paths[3].read_text());assert nd['passed'] and nd['source_target_sha256']==sha(paths[5]) and sha(paths[4])==nd['outputs']['chain_correction.npz']
    _TABLE=load(paths[1]);assert (_TABLE['TYPE']=='PSF').all();_STEPS=[]
    for identity in _TABLE['identity']:
        name=f'object_{int(identity)}/complete.json';m=ref/name;assert sha(m)==rd['objects'][name];r=json.loads(m.read_text());assert r['passed'];_STEPS.append(r['selected_step'])
    _MODEL=inference_model(joblib.load(paths[6]));assert _MODEL.converged and not np.intersect1d(_TABLE['identity'],_MODEL.training_identity).size
    _CAL,_=calibration_vectors(_TABLE,load(paths[7]),12);app=load(paths[8]);assert not np.intersect1d(_TABLE['identity'],app['identity']).size
    original=np.unique(app['field']);bounds={f['index']:json.loads((ROOT/f['config']).read_text())['bounds_deg'] for f in fields if f['index'] in original}
    _FIELDS,_EXISTING=geographic_fields(_TABLE['RA'],_TABLE['DEC'],_TABLE['cell'],bounds)
    chain=load(paths[4]);dummy=np.ones((len(original),len(_MODEL.weights),5));hierarchy=FieldPopulation(dummy,dummy,_MODEL,original)
    decoded=[hierarchy.decode(z) for z in chain['samples']];_BETA=np.array([v[1] for v in decoded]);global_logits=np.array([v[3] for v in decoded])
    _THETA={};_WEIGHTS=chain['weights'];_DRAW_IDS=chain['draw_ids']
    for field in np.unique(_FIELDS):
        if field in original:
            j=int(np.flatnonzero(original==field)[0]);theta=np.array([v[0][j] for v in decoded])
        else:
            rng=np.random.default_rng(np.random.SeedSequence([316101,12,int(field)]));local=global_logits+.35*rng.normal(size=global_logits.shape)
            theta=softmax(np.c_[local,np.zeros(len(local))],axis=1).reshape(-1,3,5)
        _THETA[int(field)]=theta
    _OUT=root/'validation_old_target'/a.cohort;_OUT.mkdir(parents=True,exist_ok=True);freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    atomic_npz(_OUT/'shared_parameters.npz',draw_ids=_DRAW_IDS,weights=_WEIGHTS,beta=_BETA,field=np.array(sorted(_THETA)),
        theta=np.stack([_THETA[x] for x in sorted(_THETA)],axis=1),identity=_TABLE['identity'],TYPE=_TABLE['TYPE'],assigned_field=_FIELDS,existing_field=_EXISTING)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        records=[]
        for r in pool.map(one,range(len(_TABLE['identity']))):records.append(r)
    csv_rows(_OUT/'summary.csv',[dict(identity=r['identity'],TYPE='PSF',role=r['role'],assigned_field=r['assigned_field'],existing_target_field=r['existing_target_field'],
        selected_step=r['selected_step'],attempts=len(r['attempts']),passed=r['passed']) for r in records])
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,objects={f'object_{r["identity"]}/complete.json':sha(_OUT/f'object_{r["identity"]}/complete.json') for r in records},
        outputs={n:sha(_OUT/n) for n in ['shared_parameters.npz','summary.csv']},cohort=a.cohort,n=len(records),all_PSF=True,
        passed=all(r['passed'] for r in records),failures=sum(not r['passed'] for r in records),references_read=False,final_outcomes_opened=False,job=os.getenv('SLURM_JOB_ID'),
        scope='Frozen original response12/calibration12/oldtarget conditional-transfer identity, retaining SED tilts and all shared population draws. No new candidate fit, reference score or final acceptance.'))


if __name__=='__main__':main()

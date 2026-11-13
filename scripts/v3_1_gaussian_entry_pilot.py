#!/usr/bin/env python3
"""Independent joint-QMC audit of fixed-class Gaussian-basis entry."""
import concurrent.futures
import json
import multiprocessing as mp
import os
import joblib
import numpy as np
from scipy.stats import norm
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA,distance_basis
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3.selection import OPTICAL
from photdis.v3_1.certified_support import CertifiedColorSupport
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_1.gaussian_entry import class_gaussian_entry
from photdis.v3_1.luminosity import luminosity_transform

_MODELS=_TABLE=_CAL=_ROWS=_PARAMETERS=_SUPPORT=_OUT=_INPUTS=_LEGACY=None


def one(task):
    context,variant,dust=task;row=_ROWS[context];folder=_OUT/f'context{context}_{variant}_dust{int(dust)}';folder.mkdir(exist_ok=True)
    marker=folder/'complete.json'
    if marker.exists():
        d=json.loads(marker.read_text());assert d['inputs']==_INPUTS
        for name,digest in d['outputs'].items():assert sha(folder/name)==digest
        return dict(path=str(marker.relative_to(_OUT)),**{k:v for k,v in d.items() if k not in ('inputs','outputs')})
    model=_MODELS[context] if dust else _MODELS[-1]
    means=np.r_[DISTANCE_BASIS_MEAN,15.];sigmas=np.r_[DISTANCE_BASIS_SIGMA,5.]
    attempts=[];previous=None;passed=False
    for draws in (8192,32768,131072,524288):
        values=[];bounds=[]
        for seed in (319701,319702):
            path=folder/f'q{draws}_seed{seed}.npz';record=path.with_suffix('.json')
            if record.exists():
                old=json.loads(record.read_text());assert old['inputs']==_INPUTS and old['sha256']==sha(path)
                data=load(path)
            else:
                value,audit=class_gaussian_entry(model,_TABLE['flux_error'][row,OPTICAL],_TABLE['transmission'][row,OPTICAL],
                    _CAL[row,OPTICAL],means,sigmas,correction=_PARAMETERS[variant],candidate=True,support=_SUPPORT,draws=draws,seed=seed)
                data=dict(alpha=value,**audit);atomic_npz(path,**data);save_json(record,dict(inputs=_INPUTS,sha256=sha(path)))
            values.append(data['alpha']);bounds.append(float(data['pruning_bound'].max()))
        first,second=values;scale=np.maximum(first,.001)
        change=None if previous is None else float(abs(first-previous).max())
        relative=None if previous is None else float((abs(first-previous)/scale).max())
        independent=float(abs(first-second).max());indrelative=float((abs(first-second)/scale).max())
        passed=change is not None and change<.0005 and independent<.0005 and relative<.01 and indrelative<.01
        attempts.append(dict(draws=draws,previous_delta=change,previous_stabilized_relative=relative,independent_delta=independent,
            independent_stabilized_relative=indrelative,max_pruning_bound=max(bounds),passed=bool(passed)))
        if passed:break
        previous=first
    cross=None
    if not dust:
        grid,curves=_LEGACY; corrected=np.array([luminosity_transform(curves[c],grid,*_PARAMETERS[variant][c]) for c in range(3)])
        basis=np.r_[distance_basis(grid),norm.pdf(grid,15,5)[None]]*trapezoid_weights(grid)
        reference=corrected@basis.T;cross=float(abs(reference-first).max());passed=passed and cross<.003
        atomic_npz(folder/'legacy_comparison.npz',reference_grid_integral=reference,joint_integral=first)
    csv_rows(folder/'attempts.csv',attempts)
    save_json(marker,dict(inputs=_INPUTS,outputs={p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.npz','.csv') or p.name.startswith('q')},
        context=context,variant=variant,matched_local_dust=dust,identity=int(_TABLE['identity'][row]),TYPE='PSF',
        selected_draws=attempts[-1]['draws'],attempts=attempts,legacy_cross_delta=cross,passed=bool(passed),job=os.getenv('SLURM_JOB_ID')))
    print('GAUSSIAN ENTRY',context,variant,dust,passed,attempts[-1],flush=True)
    return dict(path=str(marker.relative_to(_OUT)),context=context,variant=variant,matched_local_dust=dust,
        identity=int(_TABLE['identity'][row]),selected_draws=attempts[-1]['draws'],passed=bool(passed),legacy_cross_delta=cross)


def main():
    global _MODELS,_TABLE,_CAL,_ROWS,_PARAMETERS,_SUPPORT,_OUT,_INPUTS,_LEGACY
    require_compute();root=ROOT/'data/v3_1';old=ROOT/'data/v3_0';_OUT=root/'gaussian_entry_pilot';_OUT.mkdir(exist_ok=True)
    paths=[root/'candidate_selection_pilot/context_1/complete.json',old/'cohorts/application.npz',old/'uncertainty_refit/draw_12/model.joblib',
        old/'uncertainty/shared_draws.npz',old/'controls/qrf_grz.joblib',root/'selection_certificate_audit/complete.json',
        root/'luminosity_fit_refined/parameters.npz',root/'luminosity_fit_refined/complete.json',
        root/'candidate_selection_refinement/context_1/complete.json',ROOT/'scripts/v3_1_gaussian_entry_pilot.py',
        ROOT/'docs/v3_1_gaussian_entry_protocol.md']
    paths += [ROOT/'src/photdis/v3_1'/name for name in ('gaussian_entry.py','selection_dust.py','certified_support.py','exact_support.py','luminosity.py')]
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths};freeze=_OUT/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==_INPUTS
    else:save_json(freeze,_INPUTS)
    assert json.loads(paths[5].read_text())['passed']
    original=json.loads(paths[0].read_text());_ROWS=original['context_candidate_rows'];_TABLE=load(paths[1]);assert (_TABLE['TYPE'][_ROWS]=='PSF').all()
    model=inference_model(joblib.load(paths[2]));assert model.converged
    _MODELS={c:entry_model_with_local_dust(model,_TABLE['transmission'][row],_TABLE['EBV'][row]) for c,row in enumerate(_ROWS)};_MODELS[-1]=model
    _CAL,_=calibration_vectors(_TABLE,load(paths[3]),12);_SUPPORT=CertifiedColorSupport(joblib.load(paths[4]))
    fit=json.loads(paths[7].read_text());assert sha(paths[6])==fit['outputs']['parameters.npz'];parameters=load(paths[6])
    _PARAMETERS=dict(reference=np.zeros((3,2)),point=parameters['point'],**{f'bootstrap{i:02}':parameters['bootstrap'][i] for i in (0,15,31)})
    legacy=json.loads(paths[8].read_text());curves=[]
    for k in range(len(model.weights)):
        p=paths[8].parent/f'q8192_seed317101_step0.05_k{k:03}.npz';assert sha(p)==legacy['outputs'][p.name]
        data=load(p);curves.append(data['curve'])
    prior=model.weights[:,None]*model.classes;prior/=prior.sum(axis=0)[None];_LEGACY=(data['grid'],prior.T@np.array(curves))
    tasks=[(c,v,True) for c in range(3) for v in _PARAMETERS]+[(1,v,False) for v in ('reference','point')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=17,mp_context=mp.get_context('fork')) as pool:results=list(pool.map(one,tasks))
    csv_rows(_OUT/'summary.csv',results)
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,outputs={'summary.csv':sha(_OUT/'summary.csv')},
        task_markers={r['path']:sha(_OUT/r['path']) for r in results},job=os.getenv('SLURM_JOB_ID'),tasks=len(results),
        passed=all(r['passed'] for r in results),legacy_per_kernel_quadrature_passed=legacy['passed'],
        scope='Joint Gaussian-basis numerical audit, frozen within-class kernels and luminosity draws. No target fit or external reference outcome read. Integration ESS is not scientific posterior support.'))


if __name__=='__main__':main()

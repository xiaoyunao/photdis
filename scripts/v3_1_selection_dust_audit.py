#!/usr/bin/env python3
"""Quantify the local-dust entry mismatch before any broad scientific rerun."""
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
from photdis.v3.inference import distance_basis
from photdis.v3.selection import effective_errors,gaussian_component_entry
from photdis.v3.systematics import calibration_vectors
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.selection_dust import entry_model_with_local_dust

_TABLES=_MODELS=_CAL=_OUT=_INPUTS=None


def one(task):
    cohort,row=task; t=_TABLES[cohort]; model=_MODELS[cohort]
    assert t['TYPE'][row]=='PSF'
    error=effective_errors({k:v[row:row+1] for k,v in t.items()})[0]*10**(.4*_CAL[cohort][row,[0,1,3]])
    dust=entry_model_with_local_dust(model,t['transmission'][row],t['EBV'][row])
    grid=np.linspace(-15,45,601); basis=np.r_[distance_basis(grid),norm.pdf(grid,15,5)[None]]*trapezoid_weights(grid)
    alphas=[]
    for m in (model,dust):
        value=gaussian_component_entry(m,error,grid,draws=512,seed=319301)
        curve=(1-m.binary_reference_fraction)*value[0]+m.binary_reference_fraction*value[1]
        alphas.append(curve@basis.T)
    delta=alphas[1]-alphas[0]; conditional=model.weights[:,None]*model.classes; conditional/=conditional.sum(axis=0)[None]
    path=_OUT/f'{cohort}_{row:06}.npz'; atomic_npz(path,old_alpha=alphas[0],matched_alpha=alphas[1],identity=t['identity'][[row]],TYPE=t['TYPE'][[row]])
    return dict(cohort=cohort,row=row,identity=int(t['identity'][row]),EBV=float(t['EBV'][row]),
        max_component_delta=float(abs(delta).max()),max_class_delta=float(abs(conditional.T@delta).max()),
        max_reference_mixture_delta=float(abs(model.weights@delta).max()),path=path.name,sha256=sha(path))


def main():
    global _TABLES,_MODELS,_CAL,_OUT,_INPUTS
    require_compute(); old=ROOT/'data/v3_0'; root=ROOT/'data/v3_1'; _OUT=root/'selection_dust_audit'; _OUT.mkdir(exist_ok=True)
    paths=[root/'radial_catalogs/photometry.npz',old/'cohorts/application.npz',old/'sed_fit.joblib',
        old/'uncertainty_refit/draw_12/model.joblib',old/'uncertainty/shared_draws.npz',
        root/'candidate_selection_pilot/context_1/complete.json',ROOT/'scripts/v3_1_selection_dust_audit.py',
        ROOT/'src/photdis/v3_1/selection_dust.py',ROOT/'src/photdis/v3/selection.py',ROOT/'src/photdis/v3/inference.py']
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in paths}; marker=_OUT/'complete.json'
    if marker.exists():
        done=json.loads(marker.read_text()); assert done['inputs']==_INPUTS
        for name,digest in done['outputs'].items(): assert sha(_OUT/name)==digest
        return
    _TABLES=dict(radial=load(paths[0]),application=load(paths[1])); _MODELS=dict(radial=inference_model(joblib.load(paths[2])),application=inference_model(joblib.load(paths[3])))
    _CAL=dict(radial=np.zeros_like(_TABLES['radial']['flux']))
    _CAL['application'],_=calibration_vectors(_TABLES['application'],load(paths[4]),12)
    tasks=[('radial',128*c+i) for c in (0,48,112) for i in (0,31,63,95,127)]
    tasks += [('application',int(row)) for row in json.loads(paths[5].read_text())['context_candidate_rows']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=18,mp_context=mp.get_context('fork')) as pool: rows=list(pool.map(one,tasks))
    csv_rows(_OUT/'comparisons.csv',rows)
    save_json(marker,dict(inputs=_INPUTS,outputs={r['path']:r['sha256'] for r in rows}|{'comparisons.csv':sha(_OUT/'comparisons.csv')},
        job=os.getenv('SLURM_JOB_ID'),contexts=len(rows),max_component_delta=max(r['max_component_delta'] for r in rows),
        max_class_delta=max(r['max_class_delta'] for r in rows),
        scope='Matched Gaussian local-dust covariance versus original entry approximation on fixed pilot contexts. Same512Sobol nodes; grz S/N entry only. Candidate hard support and posterior impact still require matching. No original table/model overwritten.'))


if __name__=='__main__':main()

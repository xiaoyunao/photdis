#!/usr/bin/env python3
"""Allocated resumable fixed-SED native entry curves for the radial comparison."""
import argparse
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
from photdis.v3.selection import gaussian_component_entry
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3.likelihood import trapezoid_weights

_MODEL = _PROFILES = _GRID = _OUT = _INPUTS = _DRAW = _SEED = None


def one_profile(i):
    path = _OUT/f'profile_{i:04}.npz'; marker = path.with_suffix('.json')
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == _INPUTS
        assert record['sha256'] == sha(path); return record
    profile = _PROFILES[i]; errors = np.array([np.exp(profile[0]),1.,np.exp(profile[1])])
    response = gaussian_component_entry(_MODEL,errors,_GRID,draws=_DRAW,seed=_SEED)
    curve = (1-_MODEL.binary_reference_fraction)*response[0]+_MODEL.binary_reference_fraction*response[1]
    prior = _MODEL.weights[:,None]*_MODEL.classes; prior /= prior.sum(axis=0)[None,:]
    entry = prior.T@curve
    assert np.isfinite(entry).all() and (entry >= 0).all() and (entry <= 1+1e-12).all()
    atomic_npz(path,profile=profile,mu=_GRID,entry=entry)
    record = dict(inputs=_INPUTS,profile=i,sha256=sha(path),job=os.getenv('SLURM_JOB_ID'))
    save_json(marker,record); return record


def main():
    global _MODEL,_PROFILES,_GRID,_OUT,_INPUTS,_DRAW,_SEED
    require_compute(); p = argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=32); p.add_argument('--draws',type=int,default=256)
    p.add_argument('--seed',type=int,default=311701); a = p.parse_args()
    _DRAW,_SEED = a.draws,a.seed
    model_path = ROOT/'data/v3_0/sed_fit.joblib'; table_path = ROOT/'data/v3_0/selection/fit_-1_256/table.npz'
    table_record = json.loads(table_path.with_name('complete.json').read_text())
    assert table_record['passed'] and sha(table_path) == table_record['sha256']
    paths = [model_path,table_path,ROOT/'docs/v3_1_radial_protocol.md',ROOT/'scripts/v3_1_radial_selection.py',
        ROOT/'src/photdis/v3/selection.py',ROOT/'src/photdis/v3/model.py',ROOT/'src/photdis/v3_1/radial_selection.py']
    _INPUTS = {str(x.relative_to(ROOT)):sha(x) for x in paths}
    _INPUTS.update(draws=_DRAW,seed=_SEED,mu_bounds=[-25,55],mu_step=.05,fixed_sed_tilt=[0.,0.])
    _OUT = ROOT/f'data/v3_1/radial_selection/q{_DRAW}_seed{_SEED}'; _OUT.mkdir(parents=True,exist_ok=True)
    marker = _OUT/'complete.json'
    if marker.exists():
        old = json.loads(marker.read_text()); assert old['inputs'] == _INPUTS
        for name,digest in old['outputs'].items(): assert sha(_OUT/name) == digest
        return
    original = load(table_path); _PROFILES = original['profiles']; _GRID = np.linspace(-25,55,1601)
    _MODEL = inference_model(joblib.load(model_path))
    assert _MODEL.converged
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        for r in pool.map(one_profile,range(len(_PROFILES))):
            records.append(r); print('RADIAL SELECTION',len(records),len(_PROFILES),flush=True)
    curves = np.array([load(_OUT/f'profile_{i:04}.npz')['entry'] for i in range(len(_PROFILES))])
    atomic_npz(_OUT/'table.npz',profiles=_PROFILES,mu=_GRID,entry=curves)
    # Same broad bases, fixed within-class mixture, independent Sobol seed and
    # finer curve grid: compare actual integrated entry probabilities.
    prior = _MODEL.weights[:,None]*_MODEL.classes; prior /= prior.sum(axis=0)[None,:]
    rows = []; selected = np.unique(np.linspace(0,len(curves)-1,24,dtype=int))
    for i in selected:
        for shift in [-5.,0.,5.]:
            j = int(np.argmin(abs(original['shift']-shift))); shift = float(original['shift'][j])
            basis = norm.pdf(_GRID[None,:],DISTANCE_BASIS_MEAN[:,None]+shift,DISTANCE_BASIS_SIGMA[:,None])
            current = (curves[i]*trapezoid_weights(_GRID))@basis.T
            old = prior.T@original['alpha'][i,:,:,j]
            rows.append(dict(profile=int(i),shift=shift,max_absolute_delta=float(abs(current-old).max())))
    csv_rows(_OUT/'broad_entry_reproduction.csv',rows)
    maximum = max(r['max_absolute_delta'] for r in rows)
    save_json(marker,dict(inputs=_INPUTS,profiles=len(_PROFILES),job=os.getenv('SLURM_JOB_ID'),
        outputs={name:sha(_OUT/name) for name in ['table.npz','broad_entry_reproduction.csv']},
        max_broad_entry_delta=maximum,passed=maximum < .003,
        scope='Fixed nominal SED, binary, zero within-class tilt. Curves allow arbitrary radial densities. Conditional grz entry only; no sky area/morphology detection correction.',
        remaining='Direct per-object profile interpolation and finer radial-grid checks required in recovery pilot.'))
    print('RADIAL SELECTION COMPLETE',maximum,flush=True)


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Bounded numerical-only hard-support refinement with per-kernel checkpoints."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
from types import SimpleNamespace
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.inference import distance_basis
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.exact_support import ExactColorSupport
from photdis.v3_1.candidate_quadrature import bounded_component_entry

_MODEL=_SUPPORT=_CONTEXT=_OUT=_INPUTS=None


def kernel(task):
    k,draws,seed,step = task
    path = _OUT/f'q{draws}_seed{seed}_step{step:g}_k{k:03}.npz'; marker = path.with_suffix('.json')
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == _INPUTS and sha(path) == record['sha256']
        return path.name
    model = SimpleNamespace(means=_MODEL.means[[k]],covariances=_MODEL.covariances[[k]],binary_brightness_factor=_MODEL.binary_brightness_factor)
    grid = np.linspace(-15,45,round(60/step)+1)
    response,bound = bounded_component_entry(model,np.array(_CONTEXT['raw_error']),np.array(_CONTEXT['transmission']),
        np.array(_CONTEXT['calibration_mag']),grid,support=_SUPPORT,draws=draws,seed=seed)
    binary = np.array([1-_MODEL.binary_reference_fraction,_MODEL.binary_reference_fraction])
    curve = np.einsum('b,bkm->km',binary,response)[0]; upper = np.einsum('b,bkm->km',binary,bound)[0]
    basis = distance_basis(grid)*trapezoid_weights(grid)
    atomic_npz(path,grid=grid,curve=curve,pruning_bound=upper,alpha=basis@curve,alpha_pruning_bound=basis@upper)
    save_json(marker,dict(inputs=_INPUTS,sha256=sha(path),job=os.getenv('SLURM_JOB_ID')))
    return path.name


def variant(pool,draws,seed,step):
    names = list(pool.map(kernel,[(k,draws,seed,step) for k in range(len(_MODEL.weights))]))
    alpha = np.array([load(_OUT/name)['alpha'] for name in names])
    bounds = np.array([load(_OUT/name)['alpha_pruning_bound'] for name in names])
    print('CANDIDATE REFINEMENT',draws,seed,step,'bound',bounds.max(),flush=True)
    return alpha,bounds


def main():
    global _MODEL,_SUPPORT,_CONTEXT,_OUT,_INPUTS
    require_compute(); p = argparse.ArgumentParser(); p.add_argument('--context',type=int,choices=[0,1,2],required=True); a = p.parse_args()
    old = ROOT/'data/v3_1/candidate_selection_pilot'/f'context_{a.context}'
    _CONTEXT = json.loads((old/'complete.json').read_text())
    backend = ROOT/'data/v3_1/selection_backend_audit/complete.json'; assert json.loads(backend.read_text())['passed']
    model_path = ROOT/'data/v3_0/uncertainty_refit/draw_12/model.joblib'; qrf_path = ROOT/'data/v3_0/controls/qrf_grz.joblib'
    previous_path = old/('q512_seed317101_step0.05.npz' if a.context == 1 else 'q512_seed317101_step0.1.npz')
    assert sha(previous_path) == _CONTEXT['outputs'][previous_path.name]
    paths = [old/'complete.json',previous_path,backend,model_path,qrf_path,ROOT/'scripts/v3_1_candidate_refine.py',
        ROOT/'docs/v3_1_selection_refinement.md',ROOT/'src/photdis/v3_1/exact_support.py',ROOT/'src/photdis/v3_1/candidate_quadrature.py']
    _INPUTS = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    _OUT = ROOT/'data/v3_1/candidate_selection_refinement'/f'context_{a.context}'; _OUT.mkdir(parents=True,exist_ok=True)
    freeze = _OUT/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == _INPUTS
    else: save_json(freeze,_INPUTS)
    if (_OUT/'complete.json').exists():
        done = json.loads((_OUT/'complete.json').read_text()); assert done['inputs'] == _INPUTS
        for name,digest in done['outputs'].items(): assert sha(_OUT/name) == digest
        return
    _MODEL = inference_model(joblib.load(model_path)); assert _MODEL.converged
    _SUPPORT = ExactColorSupport(joblib.load(qrf_path)); previous = load(previous_path)['alpha']; records = []; passed = False
    with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:
        for draws in (2048,8192):
            first,bound1 = variant(pool,draws,317101,.05); second,bound2 = variant(pool,draws,317102,.05)
            change = float(abs(first-previous).max()); independent = float(abs(second-first).max())
            record = dict(draws=draws,previous_level_delta=change,independent_seed_delta=independent,
                max_pruning_bound=float(max(bound1.max(),bound2.max())),fine_delta=None,passed=False)
            records.append(record)
            if change < .003 and independent < .003:
                fine,bound3 = variant(pool,draws,317101,.025); record['fine_delta'] = float(abs(fine-first).max())
                record['passed'] = passed = record['fine_delta'] < .003
                if passed: break
            previous = first
    csv_rows(_OUT/'comparisons.csv',records)
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,outputs={p.name:sha(p) for p in _OUT.iterdir() if p.suffix in ['.npz','.csv'] or p.name.startswith('q')},
        context=a.context,identity=_CONTEXT['identity'],job=os.getenv('SLURM_JOB_ID'),attempts=records,passed=passed,
        selected_draws=records[-1]['draws'],scope='Same candidate cuts/model; bounded2048/8192 hard-support QMC with independent seed and fine-grid gate. Original failures retained; no population refit.'))


if __name__ == '__main__': main()

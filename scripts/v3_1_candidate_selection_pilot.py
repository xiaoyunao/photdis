#!/usr/bin/env python3
"""Allocated exact candidate-envelope selection pilot on frozen observed contexts."""
import argparse
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.baseline_v1 import support
from photdis.v3.controls import optical_magnitudes
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.selection import OPTICAL, gaussian_component_entry
from photdis.v3.inference import distance_basis
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.candidate_selection import candidate_component_entry, candidate_thresholds


def main():
    require_compute(); p = argparse.ArgumentParser(); p.add_argument('--context', type=int, choices=[0, 1, 2], required=True); a = p.parse_args()
    start = time.monotonic(); data = ROOT/'data/v3_0'
    paths = [data/'cohorts/application.npz', data/'controls/qrf_grz.joblib',
        data/'uncertainty_refit/draw_12/model.joblib', data/'uncertainty/shared_draws.npz',
        ROOT/'scripts/v3_1_candidate_selection_pilot.py', ROOT/'src/photdis/v3_1/candidate_selection.py',
        ROOT/'src/photdis/v3/selection.py', ROOT/'src/photdis/baseline_v1.py', ROOT/'docs/v3_1_candidate_selection_protocol.md']
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    out = ROOT/'data/v3_1/candidate_selection_pilot'/f'context_{a.context}'; out.mkdir(parents=True, exist_ok=True)
    freeze = out/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze, inputs)
    if (out/'complete.json').exists():
        done = json.loads((out/'complete.json').read_text()); assert done['inputs'] == inputs
        for name, digest in done['outputs'].items(): assert sha(out/name) == digest
        return
    t = load(paths[0]); qrf = joblib.load(paths[1]); model = inference_model(joblib.load(paths[2]))
    assert model.converged
    qrf.nn.n_jobs = 1
    assert qrf.dim == 2 and getattr(qrf, 'feature_schema', 'adjacent_colors') == 'adjacent_colors'
    mag, error = optical_magnitudes(t); rmag = mag[:, 1]-2.5*np.log10(t['transmission'][:, 1])
    candidate = (t['TYPE'] == 'PSF') & (t['photometric_system'] == 'DECam_DR11') & (rmag < 20) & (error.max(axis=1) < .1) & support(qrf, mag)
    assert candidate.sum() == 3540
    calibration, _ = calibration_vectors(t, load(paths[3]), 12)
    all_error = t['flux_error'][:, OPTICAL]/t['transmission'][:, OPTICAL]*10**(.4*calibration[:, OPTICAL])
    indices = np.flatnonzero(candidate); rank = np.lexsort((t['identity'][indices], np.log(all_error[indices]).mean(axis=1)))
    chosen = indices[rank[[0, len(rank)//2, len(rank)-1]]]; row = int(chosen[a.context])
    raw_error, trans, cal = t['flux_error'][row, OPTICAL], t['transmission'][row, OPTICAL], calibration[row, OPTICAL]
    effective, thresholds = candidate_thresholds(raw_error, trans, cal)
    def accepted(magnitudes):
        # Exact bounding-box short circuit; no approximated support boundary.
        x = qrf.feature_matrix(magnitudes); inside = ((x >= qrf.lo) & (x <= qrf.hi)).all(axis=1)
        answer = np.zeros(len(x), bool)
        if inside.any(): answer[inside] = support(qrf, magnitudes[inside])
        return answer
    variants = [(128, 317101, .1), (512, 317101, .1), (512, 317102, .1)]
    if a.context == 1: variants.append((512, 317101, .05))
    summaries = []; alphas = []
    for draws, seed, step in variants:
        path = out/f'q{draws}_seed{seed}_step{step:g}.npz'; marker = path.with_suffix('.json')
        if marker.exists():
            record = json.loads(marker.read_text()); assert record['inputs'] == inputs and sha(path) == record['sha256']
            value = load(path)
        else:
            grid = np.linspace(-15, 45, round(60/step)+1)
            response = candidate_component_entry(model, raw_error, trans, cal, grid, support=accepted, draws=draws, seed=seed)
            curve = (1-model.binary_reference_fraction)*response[0]+model.binary_reference_fraction*response[1]
            alpha = np.einsum('km,dm,m->kd', curve, distance_basis(grid), trapezoid_weights(grid))
            assert np.isfinite(alpha).all() and (alpha >= 0).all() and (alpha <= 1+1e-10).all()
            atomic_npz(path, grid=grid, binary_component_entry=response, alpha=alpha)
            save_json(marker, dict(inputs=inputs, sha256=sha(path))); value = dict(alpha=alpha)
        alphas.append(value['alpha']); summaries.append(dict(draws=draws, seed=seed, step=step, path=path.name))
        print('CANDIDATE ENTRY', a.context, draws, seed, step, flush=True)
    grid = np.linspace(-15, 45, 601); original = gaussian_component_entry(model, effective, grid, draws=512, seed=317103)
    curve = (1-model.binary_reference_fraction)*original[0]+model.binary_reference_fraction*original[1]
    old = np.einsum('km,dm,m->kd', curve, distance_basis(grid), trapezoid_weights(grid))
    comparisons = [dict(comparison='q128_to512', max_delta=float(abs(alphas[0]-alphas[1]).max())),
        dict(comparison='independent_q512', max_delta=float(abs(alphas[2]-alphas[1]).max())),
        dict(comparison='candidate_above_original_entry', max_delta=float(np.maximum(alphas[1]-old, 0).max()))]
    if a.context == 1: comparisons.append(dict(comparison='grid0p1_to0p05', max_delta=float(abs(alphas[3]-alphas[1]).max())))
    csv_rows(out/'comparisons.csv', comparisons)
    atomic_npz(out/'original_entry.npz', alpha=old, identity=t['identity'][[row]], TYPE=t['TYPE'][[row]])
    save_json(out/'complete.json', dict(inputs=inputs, outputs={p.name: sha(p) for p in out.iterdir() if p.name not in ['complete.json', 'inputs.json']},
        job=os.getenv('SLURM_JOB_ID'), context=a.context, row=row, identity=int(t['identity'][row]), TYPE=str(t['TYPE'][row]),
        context_candidate_rows=chosen.tolist(), candidate_count=int(candidate.sum()), raw_error=raw_error, transmission=trans,
        calibration_mag=cal, effective_error=effective, thresholds=thresholds, variants=summaries, comparisons=comparisons,
        passed=all(x['max_delta'] < .003 for x in comparisons), elapsed_seconds=time.monotonic()-start,
        scope='Numerical candidate-envelope selection pilot only; exact QRF support and original observed calibration retained. Population refit and context interpolation are not performed.'))


if __name__ == '__main__': main()

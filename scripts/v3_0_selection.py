#!/usr/bin/env python3
"""Measured, refined conditional selection response for shared target inference."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, sha, load, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.data import stable_rank
from photdis.v3.selection import effective_errors, component_entry
from photdis.v3.selection_table import profile_centers, profile_table, SelectionTable
from photdis.v3.inference import distance_basis
from photdis.v3.likelihood import trapezoid_weights

D = ROOT/'data/v3_0'
_MODEL = None


def build_profile(profile): return profile_table(_MODEL, profile, draws=128)


def direct_error(error):
    mu = np.arange(-15., 45.0001, .05)
    response = component_entry(_MODEL, error, mu, draws=512)
    curve = (1-_MODEL.binary_reference_fraction)*response[0]+_MODEL.binary_reference_fraction*response[1]
    return (curve*trapezoid_weights(mu))@distance_basis(mu).T


def main():
    global _MODEL
    require_compute(); start = time.monotonic(); parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['benchmark', 'fit'], default='fit')
    parser.add_argument('--profiles', type=int, default=64)
    parser.add_argument('--draw', type=int, default=-1)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    model_path = D/f'sed_{args.model}.joblib' if args.draw < 0 else D/'uncertainty/ensemble.joblib'
    fitted = joblib.load(model_path); fitted = fitted if args.draw < 0 else fitted[args.draw]
    _MODEL = inference_model(fitted)
    destination = D/'selection'/f'{args.model}_{args.draw}_{args.profiles}'; destination.mkdir(parents=True, exist_ok=True)
    paths = [model_path, ROOT/'scripts/v3_0_selection.py', ROOT/'src/photdis/v3/model.py',
             ROOT/'src/photdis/v3/selection.py', ROOT/'src/photdis/v3/selection_table.py',
             ROOT/'src/photdis/v3/inference.py']
    paths += [D/'cohorts'/(name+'.npz') for name in ('hyper', 'development', 'application')]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    inputs.update(draw=args.draw, requested_profiles=args.profiles)
    freeze = destination/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze, inputs)
    marker = destination/'complete.json'
    if marker.exists():
        old = json.loads(marker.read_text()); assert old['inputs'] == inputs
        assert old['sha256'] == sha(destination/'table.npz')
        if not old['passed']: raise RuntimeError('Cached selection table failed validation; refine profiles')
        return
    errors = []; identities = []
    for name in ('hyper', 'development', 'application'):
        t = load(D/'cohorts'/(name+'.npz'))
        assert (t['TYPE'] == 'PSF').all()
        errors.append(effective_errors(t)); identities.append(t['identity'])
    errors = np.concatenate(errors); identities = np.concatenate(identities)
    centers = profile_centers(errors, args.profiles)
    print('SELECTION PROFILES', len(centers), 'requested', args.profiles, flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        records = list(pool.map(build_profile, centers))
    shifts = records[0][0]
    assert all(np.array_equal(r[0], shifts) for r in records)
    table = SelectionTable(centers, shifts, np.array([r[1] for r in records], 'f4'))
    del records
    # Fixed hash subset plus extremes in absolute/relative instrumental error.
    order = np.argsort(stable_rank(identities, 302202))[:32]
    extrema = []
    for j in range(3): extrema.extend([errors[:, j].argmin(), errors[:, j].argmax()])
    checked = np.unique(np.r_[order, extrema]).astype(int)
    approximated, audit = table.evaluate(errors[checked])
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        exact = np.array(list(pool.map(direct_error, errors[checked])))
    differences = abs(approximated-exact)
    maximum = float(differences.max()); q99 = float(np.quantile(differences, .99))
    all_inside = True
    for first in range(0, len(errors), 1024):
        _, check = table.evaluate(errors[first:first+1024])
        all_inside &= bool(check['inside_profile_hull'].all())
    atomic_npz(destination/'table.npz', profiles=centers, shift=shifts, alpha=table.alpha)
    atomic_npz(destination/'validation.npz', identity=identities[checked], errors=errors[checked],
        approximate=approximated, direct=exact, profile_inside=audit['inside_profile_hull'])
    passed = maximum <= .005 and all_inside
    save_json(marker, dict(inputs=inputs, sha256=sha(destination/'table.npz'),
        validation_sha256=sha(destination/'validation.npz'), profiles=len(centers),
        construction_objects=len(errors), checked=len(checked), all_construction_inside_hull=all_inside,
        maximum_component_entry_difference=maximum, q99_component_entry_difference=q99,
        draws=128, validation_draws=512, passed=passed, elapsed_seconds=time.monotonic()-start,
        conditional_scope='Given native PSF/image/sky/reported error covariates; not an intrinsic Galactic density or effective survey area.',
        job=os.getenv('SLURM_JOB_ID')))
    print('SELECTION VALIDATION', passed, maximum, q99, flush=True)
    if not passed: raise RuntimeError('Selection interpolation requires numerical refinement; do not use for target inference')


if __name__ == '__main__': main()

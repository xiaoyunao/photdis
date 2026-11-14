#!/usr/bin/env python3
"""Produce outcome-closed external forecasts for the registered adjustment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp, softmax

from photdis.stage1 import require_compute
from photdis.v3.inference import distance_basis, kernel_loglike
from photdis.v3.likelihood import ContinuousPosterior
from photdis.v3.model import inference_model
from photdis.v3_1.radial_products import bin_integrals
from photdis.v3_1.shared_prior import geographic_fields, marginalize_native
from photdis.v3_2.giant_branch import adjust_giant_branch, reference_class_curves
from v2_1_near import ROOT, atomic_npz, load, save_json, sha
from v3_0_response import observation
from v3_2_reference_controls import external_calibration, forecast_grids, stack_view
from v3_2_reference_freeze import COMMON_ROLE, IDENTITY, MINESWEEPER_ROLE, require_native
from v3_2_tilt_inputs import verified

DECISION = ROOT/'manifests/v3_2/branch_decision.json'
_STATE = None


def forecast_level(model, obs, calibration, grid, theta, beta, weights):
    logs, _ = kernel_loglike(model, obs, grid, calibration_mag=calibration)
    base = marginalize_native(model, grid, logs, theta, beta, weights)
    binary = model.binary_reference_fraction
    native = logsumexp(logs + np.log([1-binary, binary])[:, None, None], axis=0)
    with np.errstate(divide='ignore'):
        radial = logsumexp(np.log(base['coefficient'])[..., None]
                           + np.log(distance_basis(grid))[None, :, None, :], axis=1)
    curves = logsumexp(native[:, None, :] + radial, axis=0)
    adjusted = adjust_giant_branch(grid, curves, reference_class_curves(model, grid, native),
                                   model.weights @ model.classes)
    logpdf = logsumexp(adjusted, axis=0)
    quantiles = ContinuousPosterior(grid, np.exp(logpdf)[None], 0., {}).quantiles()
    joint = bin_integrals(np.exp(adjusted), grid)
    np.testing.assert_allclose(joint.sum(), 1., rtol=0, atol=1e-9)
    good = base['joint_delta'] < 1e-8 and base['marginal_half_delta'] < .005
    return logpdf, quantiles, joint, good


def forecast_row(row):
    photo, model, calibration, grids, fields, theta, beta, weights = _STATE
    return [forecast_level(model, observation(photo, row), calibration[row], grid,
                           theta[int(fields[row])], beta, weights) for grid in grids]


def field_draws(photo, fit, forecast):
    prepared = 'data/baseline_v0/prepared.json'
    path = verified(ROOT/prepared, forecast['inputs'][prepared])
    bounds = {}
    for item in json.loads(path.read_text())['fields']:
        config = item['config']
        if config in forecast['inputs']:
            bounds[item['index']] = json.loads(
                verified(ROOT/config, forecast['inputs'][config]).read_text())['bounds_deg']
    if not set(fit['fields']).issubset(bounds):
        raise ValueError('Missing frozen population field definitions')
    fields, original = geographic_fields(photo['RA'], photo['DEC'], photo['cell'], bounds)
    if original.any():
        raise ValueError('Prospective admission overlaps an original field')
    draws = {}
    for field in np.unique(fields):
        rng = np.random.default_rng(np.random.SeedSequence([316101, 12, int(field)]))
        logits = fit['global_logits'] + .35*rng.normal(size=fit['global_logits'].shape)
        draws[int(field)] = softmax(np.c_[logits, np.zeros(len(logits))], axis=1).reshape(-1, 3, 5)
    return fields, draws


def main():
    global _STATE
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=('minesweeper', 'common_scale'))
    args = parser.parse_args()
    root = ROOT/'data/v3_2'
    prefix = args.cohort
    native, role = ('minesweeper_native', MINESWEEPER_ROLE) if prefix == 'minesweeper' else (
        'common_scale_inventory', COMMON_ROLE)
    decision = json.loads(DECISION.read_text())
    if decision['development_evaluation_complete'] is not True or decision['new_outcomes_read'] is not False:
        raise ValueError('The registered adjustment requires completed opened evaluation')
    sources = {name: verified(ROOT/name, digest) for name, digest in decision['inputs'].items()}
    native_record = ROOT/f'manifests/v3_2/{native}/complete.json'
    native_path = verified(root/native/'complete.json', sha(native_record))
    census = json.loads(native_path.read_text())
    photo_path = verified(native_path.with_name('photometry.npz'), census['outputs']['photometry.npz'])
    photo = load(photo_path)
    require_native(photo)
    controls = ROOT/f'manifests/v3_2/{prefix}_controls/complete.json'
    control = json.loads(verified(root/f'{prefix}_controls/complete.json', sha(controls)).read_text())
    if (control['cohort'], control['n'], control['references_read'], control['photometry_sha256']) != (
            role, len(photo['identity']), False, sha(photo_path)):
        raise ValueError('Frozen comparator cohort mismatch')
    opened = 'manifests/v3_2/tilt_predictions/initial/complete.json'
    forecast = json.loads(sources[opened].read_text())
    fit_marker = 'manifests/v3_2/tilt_fits/initial/candidate_free/complete.json'
    fit_path = verified(root/'tilt_fits/initial/candidate_free/complete.json', sha(sources[fit_marker]))
    fit_record = json.loads(fit_path.read_text())
    posterior_path = verified(fit_path.with_name('posterior.npz'), fit_record['outputs']['posterior.npz'])
    fit = load(posterior_path)
    if (fit_record['sampler_passed'] is not True or fit_record['n'] != 3540
            or len(fit['identity']) != 3540 or not (fit['TYPE'] == 'PSF').all()
            or np.intersect1d(fit['identity'], photo['identity']).size):
        raise ValueError('Candidate/free fit or external identity qualification failed')
    def frozen_input(name):
        return verified(ROOT/name, forecast['inputs'][name])
    model_path = frozen_input('data/v3_0/uncertainty_refit/draw_12/model.joblib')
    shared_path = frozen_input('data/v3_0/uncertainty/shared_draws.npz')
    model = inference_model(joblib.load(model_path))
    if np.intersect1d(model.training_identity, photo['identity']).size:
        raise ValueError('External identity overlaps SED fit')
    fields, theta = field_draws(photo, fit, forecast)
    grids = forecast_grids()
    beta = fit['beta']
    _STATE = photo, model, external_calibration(photo, load(shared_path)), grids, fields, theta, beta, np.full(len(beta), 1/len(beta))
    destination = root/f'{prefix}_branch_candidate'
    if destination.exists():
        raise FileExistsError(destination)
    with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('fork')) as pool:
        rows = list(pool.map(forecast_row, range(len(photo['identity']))))
    view = stack_view(rows, grids)
    destination.mkdir(exist_ok=False)
    payload = destination/'candidate.npz'
    atomic_npz(payload, **{key: photo[key] for key in IDENTITY}, **view)
    bound = (*sources.values(), DECISION, Path(__file__).resolve(), controls, native_path,
             photo_path, fit_path, posterior_path, model_path, shared_path,
             ROOT/'scripts/v3_2_reference_controls.py', ROOT/'src/photdis/v3_1/shared_prior.py')
    save_json(destination/'complete.json', dict(
        cohort=role, n=len(photo['identity']), all_PSF=True, references_read=False, pilot=False,
        photometry_sha256=sha(photo_path), decision_sha256=sha(DECISION),
        views=dict(candidate=dict(payload=str(payload.relative_to(ROOT)), sha256=sha(payload),
                                  passed=int(view['numerical_pass'].sum()))),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in bound}, job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()

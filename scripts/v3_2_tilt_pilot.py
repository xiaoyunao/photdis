#!/usr/bin/env python3
"""Audit kernel-resolved native and entry arrays before fitting any tilt arm."""
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.stage1 import require_compute
from photdis.v3.inference import DISTANCE_BASIS_MEAN as MEANS, DISTANCE_BASIS_SIGMA as SIGMAS
from photdis.v3.inference import kernel_loglike
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3.model import inference_model
from photdis.v3.selection import OPTICAL
from photdis.v3.target import mixture_response, sed_tilt_features
from photdis.v3_1.certified_support import CertifiedColorSupport
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_2.kernel_entry import kernel_entry
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_response import observation


def main():
    require_compute()
    out = Path(sys.argv[1]).resolve()
    parent = ROOT / 'data/v3_2'
    if not out.is_relative_to(parent) or out == parent or out.exists():
        raise ValueError('Choose a new output directory below data/v3_2')
    old = ROOT / 'data/v3_0'
    source = ROOT / 'data/v3_1'
    pilot = source / 'target_native_pilot'
    input_names = [pilot / 'complete.json', pilot / 'roles.npz',
                   old / 'cohorts/application.npz', old / 'uncertainty_refit/draw_12/model.joblib',
                   old / 'shared_predictions/application_refit_0/draw_12/shared_system.npz',
                   old / 'controls/qrf_grz.joblib']
    pmark = json.loads(input_names[0].read_text())
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in input_names}
    inputs[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    for name in ('src/photdis/v3_2/kernel_entry.py', 'src/photdis/v3/inference.py',
                 'src/photdis/v3_1/selection_dust.py', 'src/photdis/v3/target.py',
                 'docs/v3_2_protocol.md'):
        inputs[name] = sha(ROOT / name)
    if not pmark['passed'] or sha(input_names[1]) != pmark['outputs']['roles.npz']:
        raise ValueError('Invalid v3.1 pilot marker')
    for p in input_names[2:]:
        key = str(p.relative_to(ROOT))
        if pmark['inputs'][key] != inputs[key]:
            raise ValueError(f'Original pilot input changed: {key}')
    roles, table, cal = (load(input_names[i]) for i in (1, 2, 4))
    np.testing.assert_array_equal(roles['identity'], table['identity'])
    np.testing.assert_array_equal(cal['identity'], table['identity'])
    if not (table['TYPE'] == 'PSF').all() or not (roles['TYPE'] == 'PSF').all():
        raise ValueError('Verified native PSF required')
    model = inference_model(joblib.load(input_names[3]))
    support = CertifiedColorSupport(joblib.load(input_names[5]))
    features, _ = sed_tilt_features(model)
    prior = model.weights[:, None] * model.classes
    prior /= prior.sum(axis=0)
    rows = np.sort(np.r_[roles['pilot_rows'][roles['candidate'][roles['pilot_rows']]][:4],
                         roles['pilot_rows'][~roles['candidate'][roles['pilot_rows']]][:4]])
    if len(np.unique(rows)) != 8:
        raise ValueError('The pilot needs four observations from each admission stratum')
    out.mkdir(parents=True)
    checks = []
    for row in rows:
        identity = int(table['identity'][row])
        native = source / 'target_native' / f'object_{identity}.npz'
        marker = native.with_suffix('.json')
        record = json.loads(marker.read_text())
        if sha(native) != record['sha256'] or record['identity'] != identity or not record['passed']:
            raise ValueError(f'Invalid native record: {identity}')
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
        saved = load(native)
        grid = saved['coarse_grid']
        obs = observation(table, row)
        calib = cal['calibration_mag'][row]
        logs, _ = kernel_loglike(model, obs, grid, calibration_mag=calib)
        binary = np.array([1 - model.binary_reference_fraction, model.binary_reference_fraction])
        native_log = logsumexp(logs + np.log(binary)[:, None, None], axis=0)
        radial = norm.logpdf(grid[None, :], MEANS[:, None], SIGMAS[:, None])
        component_log = logsumexp(native_log[:, None, :] + radial[None, :, :] +
                                  np.log(trapezoid_weights(grid))[None, None, :], axis=2)
        component = np.exp(component_log - float(saved['logscale']))
        collapse = prior.T @ component
        frozen = saved['coarse_response'][0]
        relevant = np.maximum(collapse, frozen) > np.exp(-30)
        native_delta = float(np.max(abs(np.log(collapse[relevant]) - np.log(frozen[relevant]))))
        if native_delta > 1e-8:
            raise ValueError(f'Zero-tilt numerator fails identity for {identity}: {native_delta}')
        dusty = entry_model_with_local_dust(model, table['transmission'][row], table['EBV'][row])
        for admission in ('broad', 'candidate') if roles['candidate'][row] else ('broad',):
            directory = source / 'target_entry' / admission / f'object_{identity}'
            marker = directory / 'complete.json'
            entry_record = json.loads(marker.read_text())
            path = directory / f"q{entry_record['selected_draws']}_seed319711.npz"
            if sha(path) != entry_record['outputs'][path.name] or not entry_record['passed']:
                raise ValueError(f'Invalid entry record: {identity} {admission}')
            inputs[str(marker.relative_to(ROOT))] = sha(marker)
            exact = load(path)['alpha'][0]
            context = (dusty, table['flux_error'][row, OPTICAL],
                       table['transmission'][row, OPTICAL], calib[OPTICAL], MEANS, SIGMAS)
            started = time.monotonic()
            old_level, _, _ = kernel_entry(*context, support=support if admission == 'candidate' else None,
                                            draws=131072, seed=319711)
            base, bound, counts = kernel_entry(*context, support=support if admission == 'candidate' else None,
                                               draws=524288, seed=319711)
            repeat, _, _ = kernel_entry(*context, support=support if admission == 'candidate' else None,
                                        draws=524288, seed=319712)
            alpha = prior.T @ base
            errors = [float(abs(alpha - factor).max()) for factor in
                      (exact, prior.T @ old_level, prior.T @ repeat)]
            alpha_delta, level_delta, seed_delta = errors
            relative_delta = max(float(np.max(abs(alpha - comparison) / np.maximum(alpha, .001)))
                                 for comparison in (exact, prior.T @ old_level, prior.T @ repeat))
            tilt_stability = []
            for beta in (np.zeros(2), np.array([.3, 0.]), np.array([0., .3]),
                         np.array([-.3, 0.]), np.array([0., -.3])):
                _, a, _ = mixture_response(component[None], base[None], model.weights,
                                           model.classes, features, beta)
                _, b, _ = mixture_response(component[None], repeat[None], model.weights,
                                           model.classes, features, beta)
                tilt_stability.append(float(abs(a - b).max()))
            checks.append(dict(identity=identity, TYPE='PSF', admission=admission,
                               native_log_delta=native_delta, v31_alpha_delta=alpha_delta,
                               level_delta=level_delta, seed_delta=seed_delta,
                               stabilized_relative_delta=relative_delta, seconds=time.monotonic()-started,
                               max_tilt_seed_delta=max(tilt_stability), min_kernel_draws=int(counts.min()),
                               maximum_kernel_alpha=float(base.max()),
                               passed=bool(max([*errors, *tilt_stability]) < .0005 and
                                           relative_delta < .01 and (prior.T @ bound).max() < .0005 and
                                           base.max() <= 1.001 and counts.min() > 0)))
            atomic_npz(out / f'{identity}_{admission}.npz', identity=np.array([identity]),
                       TYPE=np.array(['PSF']), response=component, entry=base, previous_entry=old_level,
                       second_seed=repeat, pruning_bound=bound, draws_per_kernel=counts)
    csv_rows(out / 'checks.csv', checks)
    save_json(out / 'complete.json', dict(inputs=inputs,
              outputs={p.name: sha(p) for p in sorted(out.iterdir())},
              all_PSF=True, n=len(rows), contexts=len(checks), passed=all(r['passed'] for r in checks),
              job=os.environ.get('SLURM_JOB_ID'),
              scope='Fixed pilot of zero collapse and entry precision. No new arm fitted.'))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compute matched four-arm scores and class-conditional diagnostics."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import norm

from photdis.stage1 import require_compute
from photdis.v3.inference import distance_basis, kernel_loglike
from photdis.v3.likelihood import ContinuousPosterior, trapezoid_weights
from photdis.v3.metrics import observed_parallax, normal_reference_score
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.target import joint_coefficients
from photdis.v3_1.shared_prior import geographic_fields, marginalize_native
from photdis.v3_1.validation_scores import score_pdf
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_response import observation
from v3_1_validation_report import read_cohort
from v3_2_tilt_inputs import verified

ARMS = ('broad_zero', 'broad_free', 'candidate_zero', 'candidate_free')
_STATE = None


def diagnostic(grid, class_curve, label, evidence):
    normalized = class_curve - logsumexp(class_curve + np.log(trapezoid_weights(grid)))
    posterior = ContinuousPosterior(grid, np.exp(normalized)[None], 0., {})
    modes, masses = posterior.modes()
    modes, masses = np.asarray(modes), np.asarray(masses)
    far = modes >= 15
    far_peak = float(modes[far][np.argmax(masses[far])]) if far.any() else None
    stats = dict(label=label, median_mu=float(posterior.quantiles()[2]),
                 far_mass=float(1 - posterior.cdf(15.).sum()), far_peak_mu=far_peak)
    if evidence['parallax_eligible']:
        stats['gaia_score'] = observed_parallax(
            grid, normalized, float(evidence['p_corr']), float(evidence['gaia_parallax_error']))['log_score']
    if evidence['distance_available']:
        ref_mu = 10 + 5 * np.log10(evidence['distance_kpc'])
        sigma = 5 / np.log(10) * evidence['distance_error_kpc'] / evidence['distance_kpc']
        stats['astronn_score'] = normal_reference_score(grid, normalized, ref_mu, sigma)
    return stats


def entry_transfer_scores(model, grid, native, theta, beta, value, weights, base, evidence):
    """Reuse the fixed-grid numerator and its per-draw normalizers."""
    if value['draw_probability'].shape[0] != weights.shape[1]:
        raise ValueError('Entry transfer weights must retain population draw order')
    probability = np.einsum('ls,scd->lcd', weights, value['draw_probability'])
    np.testing.assert_allclose(probability[1], value['class_distance'], rtol=0, atol=1e-10)
    variants = {}
    if not (evidence['parallax_eligible'] or evidence['distance_available']):
        return probability, variants
    # Shift the native log normalization by one common offset to avoid
    # overflow when an object has very small marginal likelihood.
    z = value['log_evidence_per_draw']
    offset = float(z.min())
    weights_over_evidence = weights[(0, 2), :]*np.exp(offset-z)
    coefficient = np.zeros((2, len(model.weights), 5, 3))
    for start in range(0, len(beta), 256):
        stop = min(start+256, len(beta))
        draws = joint_coefficients(model, theta[start:stop], beta[start:stop])
        coefficient += np.einsum('ls,skdc->lkdc', weights_over_evidence[:, start:stop], draws)
    with np.errstate(divide='ignore'):
        radial_prior = np.log(distance_basis(grid))[None, :, None, :]
    for i, level in enumerate((0, 2)):
        with np.errstate(divide='ignore'):
            prior = logsumexp(np.log(coefficient[i])[..., None]+radial_prior, axis=1)
        lp = logsumexp(native[:, None, :]+prior, axis=(0, 1))
        if evidence['parallax_eligible']:
            gaia = observed_parallax(grid, lp, evidence['p_corr'], evidence['gaia_parallax_error'])
            variants[f'entry_{level}_gaia_score_delta'] = gaia['log_score']-base['gaia']['log_score']
            variants[f'entry_{level}_gaia_pit_delta'] = gaia['pit']-base['gaia']['pit']
        if evidence['distance_available']:
            distance = evidence['distance_kpc']
            mu = 10+5*np.log10(distance)
            sigma = 5/np.log(10)*evidence['distance_error_kpc']/distance
            score = normal_reference_score(grid, lp, mu, sigma)
            variants[f'entry_{level}_teacher_score_delta'] = score-base['references']['astronn']['log_score']
    return probability, variants


def one(row):
    photo, evidence, cal, model, arms, fields, steps, output = _STATE
    identity = int(photo['identity'][row])
    if photo['TYPE'][row] != 'PSF':
        raise ValueError('Non-PSF prediction attempted')
    ev = {k: v[row] for k, v in evidence.items()}
    label = int(ev['class_label'])
    step = steps[row]
    grid = np.linspace(-20., 50., round(70 / step) + 1)
    logs, _ = kernel_loglike(model, observation(photo, row), grid, calibration_mag=cal[row])
    coarse_grid = grid[::2] if (len(grid) - 1) % 2 == 0 else np.linspace(-20., 50., round(35 / step) + 1)
    if len(grid) == 2 * len(coarse_grid) - 1:
        coarse_logs = logs[..., ::2]
    else:
        coarse_logs, _ = kernel_loglike(model, observation(photo, row), coarse_grid, calibration_mag=cal[row])
    if photo['band_mask'][row, [0, 1]].all():
        dered = photo['flux'][row, [0, 1]] / photo['transmission'][row, [0, 1]]
        color = float(-2.5 * np.log10(dered[0] / dered[1])) if (dered > 0).all() else None
    else:
        color = None
    binary = model.binary_reference_fraction
    native = logsumexp(logs + np.log([1 - binary, binary])[:, None, None], axis=0)
    prior = model.weights[:, None] * model.classes
    prior /= prior.sum(axis=0)
    with np.errstate(divide='ignore'):
        ref_curve = logsumexp(native[:, None, :] + np.log(prior)[:, :, None], axis=0)
    ref_curve += norm.logpdf(grid, 15., 5.)
    ref_evidence = logsumexp(ref_curve + np.log(trapezoid_weights(grid))[None], axis=1)
    ref_class_probability = softmax(ref_evidence + np.log(model.weights @ model.classes))
    arrays = dict(identity=np.array([identity]), TYPE=np.array(['PSF']), grid=grid,
                  reference_class_log_curve=ref_curve, reference_class_log_evidence=ref_evidence)
    rows = []
    for arm, (fit, meta) in arms.items():
        theta = fit[int(fields[row])]
        beta = meta['beta']
        weights = np.full(len(beta), 1 / len(beta))
        value = marginalize_native(model, grid, logs, theta, beta, weights)
        coarse = marginalize_native(model, coarse_grid, coarse_logs, theta, beta, weights)
        coarse_score = score_pdf(coarse_grid, coarse['full_logpdf'],
                                 coarse['posterior'].quantiles(), coarse['class_distance'], ev)
        # Sum the original binary/kernel axes after forming the normalized cut mixture.
        coefficients = value['coefficient']
        with np.errstate(divide='ignore'):
            radial = logsumexp(np.log(coefficients)[:, :, :, None] +
                               np.log(distance_basis(grid))[None, :, None], axis=1)
        class_curve = logsumexp(native[:, None, :] + radial, axis=0)
        normalizer = logsumexp(class_curve + np.log(trapezoid_weights(grid))[None])
        class_curve -= normalizer
        lp = logsumexp(class_curve, axis=0)
        q = value['posterior'].quantiles()
        scored = score_pdf(grid, lp, q, value['class_distance'], ev)
        coarse_joint_delta = float(abs(value['class_distance'] - coarse['class_distance']).max())
        coarse_mu_delta = float(abs(q - coarse['posterior'].quantiles()).max())
        coarse_score_delta = (float(abs(scored['gaia']['log_score'] - coarse_score['gaia']['log_score']))
                              if ev['parallax_eligible'] else None)
        result = dict(identity=identity, TYPE='PSF', role=str(photo['role'][row]),
                      cell=int(photo['cell'][row]), view=arm, class_label=label,
                      candidate_admission=bool(photo['candidate_admission'][row]),
                      coarse_joint_delta=coarse_joint_delta, coarse_mu_delta=coarse_mu_delta,
                      coarse_gaia_score_delta=coarse_score_delta,
                      grid_passed=coarse_joint_delta < .003 and coarse_mu_delta < .03
                      and (coarse_score_delta is None or coarse_score_delta < .02),
                      p_giant=scored['class_probability'][1], qnear=scored['qnear'],
                      predicted_class=int(np.argmax(scored['class_probability'])),
                      class_log_score=scored.get('class', {}).get('log_score'),
                      signed_gaia_log_score=scored.get('gaia', {}).get('log_score'),
                      median_mu=float(q[2]), joint_delta=value['joint_delta'],
                      marginal_half_delta=value['marginal_half_delta'],
                      sampler_passed=meta['sampler_passed'],
                      r0=float(photo['r0'][row]), g_r=color,
                      w1_state=int(photo['band_state'][row, 4]),
                      feh=float(ev['feh']) if ev['feh_valid'] else None,
                      reference_class_logz_dwarf=float(ref_evidence[0]),
                      reference_class_logz_giant=float(ref_evidence[1]),
                      reference_class_logz_other=float(ref_evidence[2]),
                      reference_p_giant=float(ref_class_probability[1]))
        if label >= 0:
            result.update({f'oracle_{k}': v for k, v in diagnostic(grid, class_curve[label], label, ev).items()})
            result.update({f'reference_oracle_{k}': v for k, v in diagnostic(grid, ref_curve[label], label, ev).items()})
        entry, sensitivity = entry_transfer_scores(
            model, grid, native, theta, beta, value, meta['entry_weights'], scored, ev)
        result.update(sensitivity)
        arrays[arm + '_entry_class_distance'] = entry
        for c, values in enumerate(value['class_distance']):
            result.update({f'c{c}_bin{b}': float(v) for b, v in enumerate(values)})
        arrays[arm + '_class_logpdf'] = class_curve
        arrays[arm + '_draw_class_distance'] = value['draw_probability']
        rows.append(result)
    path = output / f'object_{identity}.npz'
    atomic_npz(path, **arrays)
    return rows, path.name, sha(path)


def main():
    global _STATE
    require_compute()
    source = ROOT / 'data/v3_1'
    old = ROOT / 'data/v3_0'
    inputs = {}
    photo, evidence, _, _, _ = read_cohort(source, 'field', inputs)
    if not (photo['TYPE'] == 'PSF').all():
        raise ValueError('Frozen field PSF required')
    pilot = ROOT / 'manifests/v3_2/tilt_pilot_refined/complete.json'
    roots = json.loads(pilot.read_text())['inputs']
    model_path = old / 'uncertainty_refit/draw_12/model.joblib'
    model = inference_model(joblib.load(verified(model_path, roots[str(model_path.relative_to(ROOT))])))
    shared_path = old / 'uncertainty/shared_draws.npz'
    prepared = ROOT / 'data/baseline_v0/prepared.json'
    original = load(source / 'target_native_pilot/roles.npz')
    if np.intersect1d(original['identity'], photo['identity']).size:
        raise ValueError('Prediction identities overlap population fit')
    bounds = {}
    paths = [pilot, model_path, shared_path, prepared, Path(__file__).resolve()]
    paths.extend(ROOT / f'src/photdis/{name}' for name in
                 ('v3/inference.py', 'v3/systematics.py', 'v3/likelihood.py',
                  'v3/metrics.py', 'v3_1/shared_prior.py', 'v3_1/validation_scores.py'))
    for field in json.loads(prepared.read_text())['fields']:
        if field['index'] in np.unique(original['field']):
            config = ROOT / field['config']
            bounds[field['index']] = json.loads(config.read_text())['bounds_deg']
            paths.append(config)
    fields, _ = geographic_fields(photo['RA'], photo['DEC'], photo['cell'], bounds)
    calibration, _ = calibration_vectors(photo, load(shared_path), 12)
    arms = {}
    for arm in ARMS:
        folder = ROOT / 'data/v3_2/tilt_fits/initial' / arm
        path = folder / 'complete.json'
        marker = json.loads(path.read_text())
        fit = load(verified(folder / 'posterior.npz', marker['outputs']['posterior.npz']))
        if not (fit['TYPE'] == 'PSF').all() or len(fit['identity']) != marker['n']:
            raise ValueError('Fit identity mismatch')
        lookup = {int(f): fit['theta'][:, i] for i, f in enumerate(fit['fields'])}
        for f in np.unique(fields):
            if int(f) not in lookup:
                rng = np.random.default_rng(np.random.SeedSequence([316101, 12, int(f)]))
                logits = fit['global_logits'] + .35 * rng.normal(size=fit['global_logits'].shape)
                lookup[int(f)] = softmax(np.c_[logits, np.zeros(len(logits))], axis=1).reshape(-1, 3, 5)
        transfer_path = verified(folder / 'posterior_entry_weights.npz',
                                 marker['outputs']['posterior_entry_weights.npz'])
        transfer = load(transfer_path)
        draws = len(fit['beta'])
        np.testing.assert_array_equal(transfer['draw_index'], np.arange(draws))
        entry_weights = transfer['weights']
        if entry_weights.shape != (3, draws) or not np.isfinite(entry_weights).all() or (entry_weights < 0).any():
            raise ValueError('Invalid preserved population entry weights')
        np.testing.assert_allclose(entry_weights.sum(axis=1), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(entry_weights[1], 1/draws, atol=1e-12, rtol=0)
        arms[arm] = lookup, dict(beta=fit['beta'], sampler_passed=marker['sampler_passed'],
                                entry_weights=entry_weights)
        paths.append(transfer_path)
        paths.append(path)
    reference = source / 'validation_reference_refinement/field'
    path = reference / 'complete.json'
    ref = json.loads(path.read_text())
    paths.append(path)
    if sha(shared_path) != ref['inputs'][str(shared_path.relative_to(ROOT))]:
        raise ValueError('Calibration draws differ from frozen reference')
    steps = []
    for identity in photo['identity']:
        name = f'object_{int(identity)}/complete.json'
        marker = json.loads(verified(reference / name, ref['objects'][name]).read_text())
        if marker['TYPE'] != 'PSF' or not marker['passed']:
            raise ValueError('Numerical grid has no accepted reference provenance')
        steps.append(marker['selected_step'])
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    output = ROOT / 'data/v3_2/tilt_predictions/initial'
    output.mkdir(parents=True, exist_ok=False)
    _STATE = photo, evidence, calibration, model, arms, fields, steps, output
    rows, markers = [], {}
    with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('fork')) as pool:
        for i, (scores, name, digest) in enumerate(pool.map(one, range(len(photo['identity']))), 1):
            rows.extend(scores)
            markers[name] = digest
            if i % 32 == 0:
                print(f'FIELD {i}/{len(photo["identity"])}', flush=True)
    csv_rows(output / 'objects.csv', rows)
    save_json(output / 'complete.json', dict(inputs=inputs, objects=markers,
              outputs={'objects.csv': sha(output / 'objects.csv')}, all_PSF=True,
              job=os.getenv('SLURM_JOB_ID'), n=len(markers), arms=ARMS,
              scope='Opened field development diagnostic on frozen grids. '
                    'Grid sensitivity and fresh outcomes are separate requirements.'))


if __name__ == '__main__':
    main()

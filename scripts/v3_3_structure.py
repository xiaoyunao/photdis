#!/usr/bin/env python3
"""Score a shared photometric distance with an external dynamical benchmark."""
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.stage1 import require_compute
from photdis.v3.inference import DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3.model import inference_model
from photdis.v3.selection import OPTICAL, gaussian_component_entry
from photdis.v3_1.gaussian_response import class_gaussian_logresponse
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from photdis.v3_1.structure import withdrawn_gaussian_logresponse
from photdis.v3_1.structure_posterior import common_structure_posterior
from photdis.v3_3 import class_log_likelihood
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_response import observation
from v3_2_structure_joint import validate_posterior
from v3_3_catalog import FREEZE, read_frozen
from v3_3_sed import verified

CONFIG = ROOT/'configs/v3_3_structure.json'
OUT = ROOT/'data/v3_3/structure'
_STATE = None


def class_response(curve, grid, means, sigmas):
    reference = curve + norm.logpdf(grid, 15., 5.) + np.log(trapezoid_weights(grid))
    z = logsumexp(reference, axis=1)
    mass = np.exp(reference-z[:, None])
    center = mass @ grid
    sigma = np.sqrt(np.sum(mass * (grid-center[:, None])**2, axis=1))
    full = class_gaussian_logresponse(curve, grid, means, sigmas, np.zeros((1, 3, 2)))[0]
    gaussian = withdrawn_gaussian_logresponse(z, center, sigma, means, sigmas)
    return np.stack((full, gaussian))


def one(row):
    photo, model, frequency, grid, means, sigmas, config = _STATE
    o = observation(photo, row)
    curve = class_log_likelihood(model, o, grid)
    responses = [class_response(curve[:, ::step], grid[::step], means, sigmas) for step in (1, 2)]
    responses = logsumexp(np.asarray(responses)+np.log(frequency)[None, None, :, None], axis=2)
    # The photometric entry function is identical for the two likelihoods.
    error = o['flux_error'][OPTICAL] / o['transmission'][OPTICAL]
    dust = entry_model_with_local_dust(model, o['transmission'], o['EBV'])
    entry_grid = grid[::15]
    # Include the last point so integration weights describe the entire span.
    if entry_grid[-1] != grid[-1]:
        entry_grid = np.r_[entry_grid, grid[-1]]
    radial = norm.pdf(entry_grid[None, :], means[:, None], sigmas[:, None])
    # Use tighter spacing around the narrow cluster kernel than the background
    # nodes. Evaluate entry as a smooth function interpolated from its own grid.
    narrow_grid = np.unique(np.r_[entry_grid, means[len(DISTANCE_BASIS_MEAN):]])
    f = model.binary_reference_fraction
    values = []
    for seed in (config['entry_seed'], config['entry_check_seed']):
        raw = gaussian_component_entry(dust, error, narrow_grid, draws=config['entry_draws'], seed=seed)
        point = model.weights @ ((1.-f)*raw[0]+f*raw[1])
        radial_grid = grid[(grid >= means[len(DISTANCE_BASIS_MEAN)]-.5) & (grid <= means[-1]+.5)]
        broad = (radial[:len(DISTANCE_BASIS_MEAN)]*trapezoid_weights(entry_grid)) @ np.interp(entry_grid, narrow_grid, point)
        narrow = (norm.pdf(radial_grid[None, :], means[len(DISTANCE_BASIS_MEAN):, None], sigmas[-1])
                  * trapezoid_weights(radial_grid)) @ np.interp(radial_grid, narrow_grid, point)
        values.append(np.r_[broad, narrow])
    entry = np.asarray(values)
    if not np.isfinite(entry).all() or (entry < -1e-12).any() or (entry > 1+1e-12).any():
        raise ValueError('Entry probabilities outside unit interval')
    return responses, np.clip(entry, 0, 1)


def evaluate(response, entry, radius, mu, odds, sigma):
    n = len(DISTANCE_BASIS_MEAN)
    background = logsumexp(response[:, :n], axis=1)-np.log(n)
    bg_entry = entry[:, :n].mean(axis=1)
    wm = trapezoid_weights(mu)
    wz = trapezoid_weights(odds) * norm.pdf(odds, 0., sigma)
    result = common_structure_posterior(response[None, :, n:], background[None, :],
        entry[None, :, n:], bg_entry[None, :], radius, mu, np.log(wm/wm.sum()), odds, np.log(wz/wz.sum()))
    validate_posterior(result, len(radius))
    return result


def score(mu, weights, target):
    distance = 10**((mu-10.)/5.)
    d = target['distance_kpc']
    lo, hi = (target[k] for k in ('distance_lower_error_kpc', 'distance_upper_error_kpc'))
    # Density of the external observation conditional on the photometric scale.
    sigma = np.where(distance < d, lo, hi)
    kernel = np.sqrt(2./np.pi)/(lo+hi)*np.exp(-.5*((distance-d)/sigma)**2)
    quantiles = np.interp([.025, .16, .5, .84, .975], np.cumsum(weights)-weights/2, distance)
    return dict(log_predictive_density=float(np.log(weights@kernel)), mean_kpc=float(weights@distance),
                q025_kpc=quantiles[0], q16_kpc=quantiles[1], median_kpc=quantiles[2],
                q84_kpc=quantiles[3], q975_kpc=quantiles[4],
                median_error_kpc=quantiles[2]-d, external_center_in_68=bool(quantiles[1] <= d <= quantiles[3]))


def main():
    require_compute()
    global _STATE
    config = json.loads(CONFIG.read_text())
    frozen = read_frozen()
    entry = frozen['models'][frozen['research_catalog_model']]
    model_path = verified(ROOT/entry['path'], entry['sha256'])
    model = inference_model(joblib.load(model_path))
    root = ROOT/'data/v3_1/structure'
    record_path = ROOT/'manifests/v3_1/structure/complete.json'
    record = json.loads(verified(root/'complete.json', sha(record_path)).read_text())
    photo_path = verified(root/'photometry.npz', record['outputs']['photometry.npz'])
    photo = load(photo_path)
    identity, types = photo['identity'], photo['TYPE']
    if len(identity) != 512 or len(np.unique(identity)) != len(identity) or not (types == 'PSF').all():
        raise ValueError('Registered NGC288 PSF object roster required')
    grid = np.linspace(-15., 45., 4801)
    mu = np.linspace(*(10.+5.*np.log10(config['distance_prior_kpc'])), config['member_modulus_points'])
    means = np.r_[DISTANCE_BASIS_MEAN, mu]
    widths = np.r_[DISTANCE_BASIS_SIGMA, np.full(len(mu), config['member_width_mag'])]
    odds = np.linspace(*config['spatial_log_odds_grid'])
    _STATE = photo, model, model.weights@model.classes, grid, means, widths, config
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '16')),
                                                mp_context=mp.get_context('fork')) as pool:
        rows = list(pool.map(one, range(len(identity)), chunksize=2))
    response = np.stack([r for r, _ in rows], axis=2)
    selection = np.stack([e for _, e in rows], axis=1)
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_npz(OUT/'response.npz', identity=identity, TYPE=types, mu=mu, response=response, selection=selection)
    # Access the external benchmark after persisting the model response.
    scale_path = ROOT/'configs/v3_0_independent_scale.json'
    target = json.loads(scale_path.read_text())
    score_rows = []
    n = len(DISTANCE_BASIS_MEAN)
    for branch, index in (('full', 0), ('gaussian', 1)):
        for variant, native, seed, distance_step, odds_step in (
            ('fine', 0, 0, 1, 1), ('coarse_posterior', 0, 0, 2, 2),
            ('coarse_likelihood', 1, 0, 1, 1), ('second_entry_seed', 0, 1, 1, 1)):
            ix = np.r_[np.arange(n), np.arange(n, len(means), distance_step)]
            result = evaluate(response[native, index][:, ix], selection[seed][:, ix], photo['radius_deg'],
                              mu[::distance_step], odds[::odds_step], config['central_spatial_log_odds_sigma'])
            atomic_npz(OUT/f'{branch}_{variant}.npz', identity=identity, TYPE=types, **result)
            metrics = score(result['common_mu'], result['common_mu_probability'], target)
            score_rows.append(dict(branch=branch, variant=variant, expected_members=result['mean'][0],
                                   member_sd=np.sqrt(result['covariance'][0, 0]), **metrics))
    csv_rows(OUT/'distance_score.csv', score_rows)
    paths = (FREEZE, CONFIG, model_path, record_path, photo_path, scale_path, Path(__file__).resolve(),
             ROOT/'src/photdis/v3_3.py', ROOT/'src/photdis/v3_1/structure_posterior.py')
    save_json(OUT/'complete.json', dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in paths},
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir())}, n=len(identity), all_PSF=True,
        distance_held_out_from_inference=True, known_system=True, model=frozen['research_catalog_model'],
        benchmark_distance_kpc=target['distance_kpc'], scope='One fixed SED response and fixed spatial/background model.',
        job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()

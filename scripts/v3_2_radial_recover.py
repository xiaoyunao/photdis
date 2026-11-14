#!/usr/bin/env python3
"""Recover narrow structure and selected counts from frozen v3.2 simulations."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.stage1 import require_compute
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.inference import kernel_loglike
from photdis.v3.model import inference_model
from photdis.v3.population import SharedCounts
from photdis.v3.sampling import chain_diagnostics, hmc
from photdis.v3.selection import effective_errors, gaussian_component_entry
from photdis.v3_1.radial import RadialPopulation
from photdis.v3_1.radial_products import bin_integrals
from photdis.v3_1.selection_dust import entry_model_with_local_dust
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_0_response import observation
from v3_2_radial_checkpoint import compute_rows
from v3_2_tilt_inputs import verified

MODEL_PATH = ROOT / 'data/v3_0/sed_fit.joblib'
CONTEXT_KEYS = ('context_identity', 'TYPE', 'band_mask', 'flux_error', 'transmission', 'EBV')
_MODEL = _PHOTO = _GRID = _PRIOR = _ERROR = _POPULATION = _MAPS = _SEED = None


def likelihood(row):
    obs = observation(_PHOTO, row)
    lk, _ = kernel_loglike(_MODEL, obs, _GRID)
    binary = _MODEL.binary_reference_fraction
    native = logsumexp(lk + np.log([1 - binary, binary])[:, None, None], axis=0)
    with np.errstate(divide='ignore'):
        class_log = logsumexp(native[:, None, :] + np.log(_PRIOR)[:, :, None], axis=0)
    return class_log


def entry_curve(model, observation, error, grid, prior):
    binary = model.binary_reference_fraction
    dust = entry_model_with_local_dust(model, observation['transmission'], observation['EBV'])
    entry = []
    # Entry is smooth in distance. Integrate on 0.05-mag nodes and compare
    # against a separately scrambled 0.025-mag grid before interpolation.
    for step, draws, seed in ((.05, 256, 326501), (.025, 512, 326502)):
        nodes = np.linspace(grid[0], grid[-1], round(np.ptp(grid) / step) + 1)
        raw = gaussian_component_entry(dust, error, nodes, draws=draws, seed=seed)
        curve = prior.T @ ((1 - binary) * raw[0] + binary * raw[1])
        entry.append([np.interp(grid, nodes, x) for x in curve])
    return np.asarray(entry)


def likelihood_entry(row):
    obs = observation(_PHOTO, row)
    return likelihood(row), entry_curve(_MODEL, obs, _ERROR[row], _GRID, _PRIOR)


def load_entry_model(design):
    digest = design['inputs'][str(MODEL_PATH.relative_to(ROOT))]
    return inference_model(joblib.load(verified(MODEL_PATH, digest))), digest


def entry_prior(model):
    prior = model.weights[:, None] * model.classes
    frequency = prior.sum(axis=0)
    return prior / frequency, frequency


def entry_grid():
    # 0.01 mag resolves the 0.04 mag lower width bound in the model.
    return np.linspace(-20, 50, 7001)


def verify_psf(photo):
    if not (photo['TYPE'] == 'PSF').all() or not photo['band_mask'][:, [0, 1, 3]].all():
        raise ValueError('Verified native PSF grz photometry required')


def catalog_rows(design_path, record):
    path = verified(design_path.parent / 'design.csv', record['outputs']['design.csv'])
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def cached_entry(catalog, photo, design_path, model_sha):
    root = ROOT / 'data/v3_2/radial_entry'
    record = json.loads(design_path.read_text())
    repeat, = [int(r['repeat']) for r in catalog_rows(design_path, record) if int(r['catalog']) == catalog]
    folder = root / f'repeat_{repeat:02}'
    marker_path = folder / 'complete.json'
    marker = json.loads(marker_path.read_text())
    if (marker['design_sha256'] != sha(design_path) or marker['repeat'] != repeat
            or marker['model_sha256'] != model_sha or not marker['all_PSF']
            or marker['entry_code_sha256'] != entry_signature()):
        raise ValueError('Entry cache does not match the design or integration code')
    p = folder / 'curves.npz'
    data = load(verified(p, marker['outputs']['curves.npz']))
    for key in CONTEXT_KEYS:
        np.testing.assert_array_equal(data[key], photo[key], err_msg=key)
    np.testing.assert_array_equal(data['mu'], _GRID)
    entry = data['entry']
    if (entry.shape != (len(photo['TYPE']), 2, 3, len(_GRID))
            or not np.isfinite(entry).all() or entry.min() < -1e-12 or entry.max() > 1+1e-12):
        raise ValueError('Invalid cached entry probability array')
    return entry, marker_path


def entry_signature():
    # Changes to the local-dust response or quadrature invalidate a cache;
    # unrelated recovery/report code changes do not.
    import hashlib
    import inspect
    module_paths = ('v3/selection.py', 'v3_1/selection_dust.py')
    functions = (entry_curve, entry_prior, entry_grid)
    contents = ''.join(inspect.getsource(f) for f in functions)
    contents += ''.join(sha(ROOT / 'src/photdis' / p) for p in module_paths)
    return hashlib.sha256(contents.encode()).hexdigest()



def chain(index):
    fit = _MAPS[index % len(_MAPS)]
    return hmc(_POPULATION.objective, fit['map'], fit['metric'],
               draws=512, warmup=512, seed=_SEED + index, target_acceptance=.9)


def moments(population, states, seed):
    parameters = np.empty((len(states), 3))
    local_prior = []
    background_prior = []
    for i, state in enumerate(states):
        theta, eta, location, width = population.decode(state)
        local = ((theta.sum(axis=1) * eta)[:, None] * norm.pdf(population.mu, location, width)
                 if population.local else np.zeros((3, len(population.mu))))
        local_prior.append(local)
        background_prior.append(population.prior_density(state) - local)
        parameters[i] = location, width, theta.sum(axis=1) @ eta
    # Realized structure membership and joint class-distance counts share
    # the same object label draw, including background/structure correlation.
    joint_count = SharedCounts(np.arange(len(states)), 36, seed=seed)
    batch = 16
    for start in range(0, len(population.response), batch):
        end = start + batch
        probabilities = []
        response = population.response[start:end]
        for background, local in zip(background_prior, local_prior):
            masses = np.concatenate(
                [bin_integrals(response * component, population.mu).reshape(len(response), 18)
                 for component in (background, local)], axis=1)
            probabilities.append(masses / masses.sum(axis=1, keepdims=True))
        p = np.asarray(probabilities)
        joint_count.add(p, np.arange(len(states)))
    result = joint_count.result()
    collapsed = result['count_draws'][:, :18] + result['count_draws'][:, 18:]
    aggregation = np.concatenate([np.eye(18), np.eye(18)], axis=1)
    members_mean = result['conditional_means'][:, 18:].sum(axis=1)
    return parameters, collapsed, result['count_draws'][:, 18:].sum(axis=1), dict(
        expected_selected_members=members_mean,
        mean=aggregation @ result['mean'], covariance=aggregation @ result['covariance'] @ aggregation.T,
        marginal_half_delta=result['marginal_half_delta'])


def summarize(draws, truth, name):
    lo, median, hi = map(float, np.quantile(draws, [.025, .5, .975]))
    return dict(metric=name, truth=float(truth), median=median,
                lower95=lo, upper95=hi, interval_length=hi-lo, covered95=bool(lo <= truth <= hi))


def main():
    global _MODEL, _PHOTO, _GRID, _PRIOR, _ERROR, _POPULATION, _MAPS, _SEED
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog', type=int)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--cached-entry', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'data/v3_2/radial_design'
    marker = root / 'complete.json'
    design = json.loads(marker.read_text())
    if not 0 <= args.catalog < design['n_catalogs']:
        raise ValueError('Catalog outside frozen design')
    photo = root / f'photometry_{args.catalog:03}.npz'
    truth_path = root / f'truth_{args.catalog:03}.npz'
    _PHOTO = load(verified(photo, design['outputs'][photo.name]))
    _MODEL, model_sha = load_entry_model(design)
    verify_psf(_PHOTO)
    _PRIOR, frequency = entry_prior(_MODEL)
    _ERROR = effective_errors(_PHOTO)
    _GRID = entry_grid()
    output = ROOT / 'data/v3_2/radial_recovery' / f'catalog_{args.catalog:03}'
    if output.exists():
        raise FileExistsError(output)
    cache_marker = None
    if args.cached_entry:
        levels, cache_marker = cached_entry(args.catalog, _PHOTO, marker, model_sha)
    sources = [marker, photo, truth_path, MODEL_PATH, Path(__file__).resolve(),
               ROOT / 'scripts/v3_2_radial_checkpoint.py']
    if cache_marker is not None:
        sources.append(cache_marker)
    sources += [ROOT / 'src/photdis' / p for p in ('v3_1/radial.py', 'v3_1/selection_dust.py',
                'v3/inference.py', 'v3/selection.py', 'v3/population.py', 'v3/sampling.py')]
    checkpoint_meta = dict(catalog=args.catalog, cached_entry=args.cached_entry,
                           inputs={str(p.relative_to(ROOT)): sha(p) for p in sources},
                           entry_signature=entry_signature())
    checkpoint = output.parent/'row_checkpoints'/output.name
    if args.cached_entry:
        result = compute_rows(checkpoint, checkpoint_meta, _PHOTO['identity'], likelihood,
                              workers=args.workers, keys=('log_likelihood',))
        logs = np.stack([value[0] for value in result])
    else:
        result = compute_rows(checkpoint, checkpoint_meta, _PHOTO['identity'], likelihood_entry,
                              workers=args.workers, keys=('log_likelihood', 'entry'))
        logs, levels = map(np.stack, zip(*result))
    output.mkdir(parents=True, exist_ok=False)
    response = np.exp(logs - logs.max(axis=(1, 2), keepdims=True))
    entry_delta = float(abs(levels[:, 1] - levels[:, 0]).max())
    atomic_npz(output / 'curves.npz', identity=_PHOTO['identity'], TYPE=_PHOTO['TYPE'],
               mu=_GRID, log_likelihood=logs, entry=levels)
    rows, gates = [], []
    fitted = {}
    for arm in ('broad', 'local'):
        local = arm == 'local'
        _POPULATION = RadialPopulation(_GRID, response, levels[:, 1], frequency, local=local)
        previous = RadialPopulation(_GRID[::2], response[..., ::2], levels[:, 0, :, ::2],
                                    frequency, local=local)
        optima = _POPULATION.fit()
        joblib.dump(optima, output / f'{arm}_map.joblib')
        optima = [o for o in optima if o['gradient_max'] < 1e-3 and np.isfinite(o['objective'])]
        if not optima:
            gates.append(dict(arm=arm, sampler_pass=False, status='no_converged_map'))
            continue
        target = optima[0]
        _MAPS = [target]
        for o in optima[1:]:
            if len(_MAPS) < 4 and o['objective'] < target['objective'] + 10 and all(
                    np.linalg.norm(o['map'] - p['map']) > .05 for p in _MAPS):
                _MAPS.append(o)
        _SEED = 327000 + 20 * args.catalog + (10 if local else 0)
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('fork')) as pool:
            chains = list(pool.map(chain, range(4)))
        samples = np.stack([c['samples'] for c in chains])
        checks = chain_diagnostics(samples)
        states = samples.transpose(1, 0, 2).reshape(-1, samples.shape[-1])
        divergences = sum(c['divergent'] for c in chains)
        passed = checks['max_rhat'] < 1.05 and checks['minimum_ess'] >= 100 and divergences == 0
        # Do not let a common selection log shift stand in for response error.
        stride = max(1, len(states) // 64)
        shifts = np.array([previous.objective(z)[0] - _POPULATION.objective(z)[0]
                           for z in states[::stride]])
        gates.append(dict(arm=arm, sampler_pass=passed, status='sampled',
                          entry_pass=entry_delta < .003,
                          absolute_objective_pass=float(abs(shifts).max()) < .05,
                          relative_objective_pass=float(np.ptp(shifts)) < .05,
                          max_rhat=checks['max_rhat'], min_ess=checks['minimum_ess'],
                          divergences=divergences, max_entry_curve_delta=entry_delta,
                          integration_objective_shift_max=float(abs(shifts).max()),
                          integration_relative_shift_span=float(np.ptp(shifts))))
        atomic_npz(output / f'{arm}_chains.npz', samples=samples, rhat=checks['rhat'],
                   ess=checks['ess'], objective_shift=shifts)
        fitted[arm] = moments(_POPULATION, states, _SEED)
        coarse = moments(previous, states[::stride], _SEED)
        fine = moments(_POPULATION, states[::stride], _SEED)
        expected_delta = float(abs(coarse[3]['mean'] - fine[3]['mean']).max())
        membership_delta = float(abs(coarse[3]['expected_selected_members'] -
                                     fine[3]['expected_selected_members']).max())
        gates[-1].update(maximum_expected_bin_count_delta=expected_delta,
                         maximum_expected_member_count_delta=membership_delta,
                         fixed_draw_count_pass=max(expected_delta, membership_delta) < .5)
    # Open synthetic truth only after constructing both posteriors.
    truth = load(verified(truth_path, design['outputs'][truth_path.name]))
    np.testing.assert_array_equal(truth['identity'], _PHOTO['identity'])
    np.testing.assert_array_equal(truth['TYPE'], _PHOTO['TYPE'])
    bins = np.searchsorted(DISTANCE_MU_BOUNDARIES, truth['mu'], side='right')
    selected = np.bincount(truth['class_label'] * 6 + bins, minlength=18)
    for arm, (params, counts, members, covariance) in fitted.items():
        atomic_npz(output / f'{arm}_draws.npz', parameters=params, counts=counts, members=members, **covariance)
        rows.extend(dict(arm=arm, **summarize(counts[:, j], selected[j], f'count_{j//6}_{j%6}'))
                    for j in range(18))
        member_count = int(truth['local'].sum())
        if arm == 'local':
            rows.append(dict(arm=arm, **summarize(members, member_count, 'selected_members')))
        if arm == 'local':
            parameter_truth = (float(truth['location']), float(truth['width']),
                               truth['theta'].sum(axis=1) @ truth['eta'])
            for j, name in enumerate(('mu_location', 'mu_width', 'pre_entry_fraction')):
                rows.append(dict(arm=arm, **summarize(params[:, j], parameter_truth[j], name)))
    csv_rows(output / 'recovery.csv', rows)
    csv_rows(output / 'numerical.csv', gates)
    save_json(output / 'complete.json', dict(catalog=args.catalog, all_PSF=True,
        n=len(_PHOTO['identity']), inputs=checkpoint_meta['inputs'],
        outputs={p.name: sha(p) for p in sorted(output.iterdir())}, job=os.getenv('SLURM_JOB_ID'),
        fitted_arms=sorted(fitted), row_checkpoint_sha256=sha(checkpoint/'inputs.json'), scope='Fixed-context synthetic recovery; sampler and numerical flags retained.'))


if __name__ == '__main__':
    main()

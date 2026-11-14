#!/usr/bin/env python3
"""Freeze controlled narrow-structure catalogs before posterior fitting."""
from itertools import product
import json
import os
from pathlib import Path

import joblib
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.inference import DISTANCE_BASIS_MEAN as MEANS, DISTANCE_BASIS_SIGMA as SIGMAS
from photdis.v3.model import inference_model
from photdis.v3.selection import OPTICAL
from photdis.v3.simulation import empirical_library
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_2_tilt_inputs import verified

FRACTIONS = (.1, .3, .6, .9)
WIDTHS = (.08, .25)
BACKGROUNDS = {'near': [.45, .3, .15, .08, .02], 'deep': [.05, .10, .35, .35, .15]}
FIXED_LOCATION = float(10 + 5 * np.log10(4.))
N = 128
SOURCES = ('model', 'empirical')


def generate(contexts, model, library, source, theta, eta, location, width, rng, index, *, context_seed):
    """Draw a catalog conditional on entry and each fixed context."""
    context_rows = np.random.default_rng(context_seed).choice(len(contexts['identity']), N, replace=False)
    mask = contexts['band_mask'][context_rows].copy()
    error = contexts['flux_error'][context_rows].copy()
    trans = contexts['transmission'][context_rows].copy()
    ebv = contexts['EBV'][context_rows]
    if not (mask[:, OPTICAL].all() and (contexts['TYPE'][context_rows] == 'PSF').all()):
        raise ValueError('Each sampled context must have verified PSF and measured grz')
    n_kernel = len(model.weights)
    prior = model.weights[:, None] * model.classes
    prior /= prior.sum(axis=0)
    factors = np.linalg.cholesky(model.covariances[:, :6, :6])
    pools = [np.flatnonzero((library['class_label'] == c) & library['mask'][:, OPTICAL].all(axis=1))
             for c in range(3)] if source == 'empirical' else None
    if pools is not None and any(not len(a) for a in pools):
        raise ValueError('All three empirical PSF classes need grz support')
    if pools is not None:
        # Hold each context's usable band mask fixed through rejection.
        pools_by_row = [[pool[library['mask'][pool][:, used].all(axis=1)] for pool in pools]
                        for used in mask]
        if any(not len(pool) for row in pools_by_row for pool in row):
            raise ValueError('Empirical PSF sources do not cover a sampled band mask')
    measured = np.full((N, 6), np.nan)
    mu_out = np.empty(N)
    classes = np.empty(N, dtype=np.int8)
    is_local = np.empty(N, dtype=bool)
    template = np.full(N, -1, dtype=np.int64)
    kernels = np.full(N, -1, dtype=np.int64)
    trials = np.zeros(N, dtype=np.int64)
    for i in range(N):
        used = mask[i]
        ext = -2.5 * np.log10(trans[i, used])
        for _ in range(200):
            size = 64
            choice = rng.choice(15, size=size, p=theta.ravel())
            c, background = np.divmod(choice, 5)
            local = rng.random(size) < eta[c]
            mu = rng.normal(np.where(local, location, MEANS[background]),
                            np.where(local, width, SIGMAS[background]))
            binary = rng.random(size) < model.binary_reference_fraction
            selected = np.empty(size, dtype=np.int64)
            for label in range(3):
                take = c == label
                if pools is None:
                    selected[take] = rng.choice(n_kernel, size=int(take.sum()), p=prior[:, label])
                else:
                    selected[take] = rng.choice(pools_by_row[i][label], size=int(take.sum()))
            if pools is None:
                means = model.means[selected, :6]
                flux = means + np.einsum('nij,nj->ni', factors[selected], rng.normal(size=(size, 6)))
                perturb = .4 * np.log(10) * .1 * ext if ebv[i] > 0 else np.zeros_like(ext)
                flux[:, used] += means[:, used] * perturb * rng.normal(size=(size, 1))
            else:
                flux = library['flux'][selected]
            amplitude = 10 ** (-.4 * (mu - 10)) * np.where(binary, model.binary_brightness_factor, 1.)
            observed = flux[:, used] * amplitude[:, None] * trans[i, used]
            observed += rng.normal(size=observed.shape) * error[i, used]
            optical_local = np.flatnonzero(np.isin(np.flatnonzero(used), OPTICAL))
            good = np.flatnonzero((observed[:, optical_local] >= 5 * error[i, OPTICAL]).all(axis=1))
            if not len(good):
                trials[i] += size
                continue
            j = good[0]
            trials[i] += j + 1
            measured[i, used] = observed[j]
            mu_out[i], classes[i], is_local[i] = mu[j], c[j], local[j]
            if pools is None:
                kernels[i] = selected[j]
            else:
                template[i] = library['identity'][selected[j]]
            break
        else:
            raise RuntimeError(f'Conditional entry did not produce object {i} in catalog {index}')
    rows = {k: contexts[k][context_rows] for k in ('EBV', 'photometric_system', 'field', 'TYPE')}
    identity = -(920000000000000000 + index * N + np.arange(N, dtype=np.int64))
    photo = dict(**rows, identity=identity, context_identity=contexts['identity'][context_rows],
                 flux=measured, flux_error=np.where(mask, error, np.nan),
                 transmission=trans, band_mask=mask)
    truth = dict(identity=identity, TYPE=photo['TYPE'], mu=mu_out, class_label=classes,
                 local=is_local, theta=theta, eta=eta, location=location, width=width,
                 template_identity=template, model_kernel=kernels, trials=trials)
    return photo, truth


def settings(model):
    freq = model.weights @ model.classes
    for source, fraction, width, background, repeat in product(
            SOURCES, FRACTIONS, WIDTHS, BACKGROUNDS, range(8)):
        yield dict(source=source, design='fixed', fraction=fraction, width=width,
                   background=background, repeat=repeat), freq[:, None] * BACKGROUNDS[background], np.full(3, fraction), FIXED_LOCATION
    from photdis.v3_1.radial import RadialPopulation
    dummy = RadialPopulation(np.linspace(-20, 50, 71), np.ones((1, 3, 71)),
                             np.ones((1, 3, 71)), freq, local=True)
    for repeat in range(32):
        latent = np.random.default_rng(326800 + repeat).normal(size=dummy.dimension)
        theta, eta, location, width = dummy.decode(latent)
        yield dict(source='model', design='prior_predictive', fraction=float(theta.sum(axis=1) @ eta),
                   width=width, background='prior', repeat=repeat), theta, eta, location


def main():
    require_compute()
    old = ROOT / 'data/v3_0'
    paths = [old / 'sed_fit.joblib', old / 'empirical_transfer_bank.npz', old / 'cohorts/application.npz']
    provenance = ROOT / 'manifests/v3_1/radial_catalogs/complete.json'
    selection = ROOT / 'manifests/v3_1/radial_selection/q256_seed311701/complete.json'
    inputs = json.loads(provenance.read_text())['inputs']
    inputs.update(json.loads(selection.read_text())['inputs'])
    model = inference_model(joblib.load(verified(paths[0], inputs[str(paths[0].relative_to(ROOT))])))
    bank, context = [load(verified(p, inputs[str(p.relative_to(ROOT))])) for p in paths[1:]]
    for table in (bank, context):
        if 'TYPE' not in table or not (table['TYPE'] == 'PSF').all():
            raise ValueError('Frozen input violates the native PSF rule')
    library = empirical_library(bank, seed=311501)
    # Use the same context pool for both source arms. Eligibility involves
    # only native measurement masks and empirical SED band availability.
    context_mask = context['band_mask']
    supported = np.zeros(len(context_mask), dtype=bool)
    for pattern in np.unique(context_mask, axis=0):
        if not pattern[OPTICAL].all():
            continue
        available = library['mask'][:, pattern].all(axis=1)
        if all((available & (library['class_label'] == c)).any() for c in range(3)):
            supported |= (context_mask == pattern).all(axis=1)
    excluded = int((~supported).sum())
    if int(supported.sum()) < N:
        raise ValueError('Too few native contexts with common SED coverage')
    context = {name: values[supported] for name, values in context.items()}
    output = ROOT / 'data/v3_2/radial_design'
    output.mkdir(parents=True, exist_ok=False)
    inventory, products = [], {}
    for i, (case, theta, eta, location) in enumerate(settings(model)):
        photo, truth = generate(context, model, library, case['source'], theta, eta, location,
                                case['width'], np.random.default_rng(326100+i), i,
                                context_seed=326900 + case['repeat'])
        for kind, values in [('photometry', photo), ('truth', truth)]:
            path = output / f'{kind}_{i:03}.npz'
            atomic_npz(path, **values)
            products[path.name] = sha(path)
        inventory.append(dict(catalog=i, **case, n=N, true_location=location,
                              selected_members=int(truth['local'].sum()), attempts=int(truth['trials'].sum())))
        if i % 32 == 0:
            print('CATALOG', i, flush=True)
    csv_rows(output / 'design.csv', inventory)
    products['design.csv'] = sha(output / 'design.csv')
    paths += [provenance, selection, Path(__file__).resolve(), ROOT / 'docs/v3_2_radial_protocol.md',
              ROOT / 'src/photdis/v3_1/radial.py', ROOT / 'src/photdis/v3/model.py',
              ROOT / 'src/photdis/v3/simulation.py']
    save_json(output / 'complete.json', dict(
        inputs={str(p.relative_to(ROOT)): sha(p) for p in paths}, outputs=products,
        all_PSF=True, source_PSF_exclusions=0, missing_TYPE=0, n_catalogs=len(inventory),
        unsupported_context_masks=excluded, common_context_pool=len(context['identity']),
        job=os.getenv('SLURM_JOB_ID'),
        scope='Synthetic catalogs conditional on fixed native PSF contexts. Empirical teacher scales are not new absolute truth.'))


if __name__ == '__main__':
    main()

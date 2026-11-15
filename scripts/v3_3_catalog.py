#!/usr/bin/env python3
"""Export the complete registered photometric roster under frozen models."""
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3.likelihood import normalize_joint, trapezoid_weights
from photdis.v3.model import inference_model
from photdis.v3_3 import class_log_likelihood, frozen_qrf, reference_log_curves
from v2_1_near import ROOT, atomic_npz, load, save_json, sha
from v3_0_response import observation
from v3_3_catalog_inputs import CONFIG, OUTPUT
from v3_3_sed import verified

FREEZE = ROOT/'manifests/v3_3/model_freeze.json'
EXPORT = OUTPUT.parent/'posteriors'
_STATE = None


def joint_summary(mu, likelihood, class_prior):
    curve = reference_log_curves(likelihood, mu, class_prior)
    p, coarse = (normalize_joint(mu[::step], curve[:, ::step]) for step in (1, 2))
    dp, dc = p.distance_bins(), coarse.distance_bins()
    class_p = p.component_mass()
    np.testing.assert_allclose(dp.sum(), 1., atol=1e-11, rtol=0)
    np.testing.assert_allclose(dp.sum(axis=1), class_p, atol=1e-11, rtol=0)
    qnear = p.cdf(15.).sum()
    np.testing.assert_allclose(dp[:, :3].sum(), qnear, atol=1e-11, rtol=0)
    distance = 10**((mu-10.)/5.)
    density_d = p.joint_density.sum(axis=0) * 5./(np.log(10.)*distance)
    near = distance < 10.
    # Search each reporting branch in physical distance, retaining boundary maxima.
    modes = np.asarray([distance[use][density_d[use].argmax()] for use in (near, ~near)])
    modes[np.asarray([qnear, 1.-qnear]) < 1e-12] = np.nan
    prob_delta = abs(dp-dc).max()
    quantile_delta = abs(p.quantiles()-coarse.quantiles()).max()
    edge = np.asarray([p.cdf(mu[1]).sum(), (class_p-p.cdf(mu[-2])).sum()])
    rel_loglike = likelihood - likelihood.max()
    joint_density = p.joint_density.astype('f4')
    marginal_density = joint_density.sum(axis=0)
    np.testing.assert_allclose(marginal_density@trapezoid_weights(mu), 1., atol=1e-6, rtol=0)
    return dict(joint_density_per_mu=joint_density, distance_marginal_per_mu=marginal_density,
                relative_class_loglike=rel_loglike.astype('f4'), class_probability=class_p,
                class_distance_bins=dp, distance_mode_kpc=modes, qnear=qnear, qfar=1.-qnear,
                distance_quantiles_kpc=10**((p.quantiles()-10.)/5.),
                coarse_probability_delta=prob_delta, coarse_quantile_delta_mag=quantile_delta,
                edge_mass=edge, numerical_flag=prob_delta > .003 or quantile_delta > .03 or edge.max() > 1e-5)


def infer(index):
    photo, models, priors, grid = _STATE
    o = observation(photo, index)
    return {key: joint_summary(grid, class_log_likelihood(model, o, grid), priors[key])
            for key, model in models.items()}


def summarize(output, measured):
    return dict(n=len(output['qnear']), numerical_flags=int(output['numerical_flag'].sum()),
                expected_near=float(output['qnear'].sum()), expected_far=float(output['qfar'].sum()),
                expected_by_class_distance=output['class_distance_bins'].sum(axis=0).tolist(),
                qnear_gt_095=int((output['qnear'] > .95).sum()),
                measured_band_counts=measured.sum(axis=0).tolist())


def main():
    require_compute()
    frozen = json.loads(FREEZE.read_text())
    input_marker = OUTPUT/'complete.json'
    census = json.loads(verified(input_marker, frozen['catalog_inputs_sha256']).read_text())
    config = json.loads(verified(CONFIG, census['inputs'][str(CONFIG.relative_to(ROOT))]).read_text())
    grid = np.linspace(config['grid']['min_mu'], config['grid']['max_mu'], config['grid']['fine_points'])
    models = {key: joblib.load(verified(ROOT/entry['path'], entry['sha256']))
              for key, entry in frozen['models'].items()}
    qrf = models.pop('qrf')
    priors = {key: model.weights @ model.classes for key, model in models.items()}
    models = {key: inference_model(model) for key, model in models.items()}
    global _STATE
    EXPORT.mkdir(parents=True, exist_ok=False)
    counts = {}
    for name, digest in census['outputs'].items():
        photo = load(verified(OUTPUT/name, digest))
        field = Path(name).stem
        n = census['census'][field]['admitted']
        if len(photo['identity']) != n or len(np.unique(photo['identity'])) != n or not (photo['TYPE'] == 'PSF').all():
            raise ValueError(f'Roster or morphology mismatch for {field}')
        _STATE = photo, models, priors, grid
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '16')),
                                                    mp_context=mp.get_context('fork')) as pool:
            result = list(pool.map(infer, range(n), chunksize=8))
        counts[field] = {}
        for key in models:
            out = {param: np.asarray([item[key][param] for item in result]) for param in result[0][key]}
            atomic_npz(EXPORT/f'{field}_{key}.npz', identity=photo['identity'], TYPE=photo['TYPE'],
                       mu=grid, class_prior=priors[key], **out)
            counts[field][key] = summarize(out, photo['band_mask'])
        mags, errors = optical_magnitudes(photo)
        q = frozen_qrf(qrf, mags, errors, photo['identity'])
        mass, edges = q['mass'], q['mu_edges']
        qnear = np.sum(mass*np.clip((15.-edges[:, :-1])/np.diff(edges, axis=1), 0, 1), axis=1)
        atomic_npz(EXPORT/f'{field}_qrf.npz', identity=photo['identity'], TYPE=photo['TYPE'], mass=mass,
                   mu_edges=edges, qnear=qnear, supported=q['supported'],
                   distance_quantiles_kpc=q['distance'])
        counts[field]['qrf'] = dict(n=n, expected_near=float(qnear.sum()),
                                   supported=int(q['supported'].sum()))
    save_json(EXPORT/'complete.json', dict(all_PSF=True, n=census['n'], counts=counts,
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (FREEZE, input_marker, Path(__file__).resolve(),
                ROOT/'src/photdis/v3_3.py')}, outputs={p.name: sha(p) for p in sorted(EXPORT.iterdir())},
        selection='Identical hashed input roster for each model. Scope is the registered v3.0 input envelope.',
        representation='Joint and marginal density per distance modulus. For density per kpc multiply by 5/(ln(10)*d_kpc).',
        loglike='One additive constant shared between all classes and grid points of an object.',
        branches='Mass below and above 10 kpc. Mode is the density-per-kpc maximum in each branch.',
        job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()

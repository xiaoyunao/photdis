#!/usr/bin/env python3
"""Generate the three frozen comparator forecasts from external photometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax

from photdis.baseline_v1 import integrate
from photdis.stage1 import require_compute
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.controls import optical_magnitudes
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.inference import kernel_loglike
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3_1.admission import candidate_admission
from photdis.v3_1.shared_prior import geographic_fields, marginalize_native
from v2_1_near import ROOT, atomic_npz, load, save_json, sha
from v3_0_response import observation
from v3_1_luminosity_guard import predict
from v3_1_validation_predict import class_loglike
from v3_2_reference_freeze import COMMON_ROLE, IDENTITY, MINESWEEPER_ROLE, ROLE, require_native
from v3_2_tilt_inputs import verified

_STATE = None


def qrf_prediction(model, table, *, batch=128):
    ids = table['identity']
    use = ~np.isin(ids, model.training_identity) & table['band_mask'][:, [0, 1, 3]].all(axis=1)
    for name in ('flux', 'flux_error', 'transmission'):
        values = table[name][:, [0, 1, 3]]
        use &= (np.isfinite(values) & (values > 0)).all(axis=1)
    n = len(ids)
    edges = np.full((n, len(model.edges)), np.nan)
    mass = np.full((n, len(model.edges)-1), np.nan)
    quantiles = np.full((n, 5), np.nan)
    supported = np.zeros(n, bool)
    take = np.flatnonzero(use)
    for start in range(0, len(take), batch):
        j = take[start:start+batch]
        mags, errors = optical_magnitudes({k: table[k][j]
                                          for k in ('flux', 'flux_error', 'transmission', 'band_mask')})
        zero = np.zeros((len(j), 2))
        value = integrate(model, np.c_[mags, zero], np.c_[errors, zero], ids[j], draws=128)['wide']
        edges[j] = mags[:, 0, None]-model.edges[::-1]
        p = value['pdf'][:, ::-1].astype(float)
        mass[j] = p/p.sum(axis=1)[:, None]
        quantiles[j], supported[j] = value['mu'], value['supported']
    return dict(mu_edges=edges, mass=mass, mu_quantiles=quantiles, available=use,
                numerical_pass=use, in_qrf_support=supported,
                qrf_training_overlap=np.isin(ids, model.training_identity))


def continuous_views(row):
    photo, nominal, response, cal, grids, fields, theta, beta, weights = _STATE
    obs = observation(photo, row)
    all_values = {'nominal_reference': [], 'old_target': []}
    for g in grids:
        # Frozen nominal likelihood with the published N(15,5) reference
        # prior in distance modulus and its own SED class frequencies.
        l = class_loglike(nominal, obs, g, np.zeros(6))
        v = predict(l, g, nominal.weights@nominal.classes, np.zeros((1, 3, 2)))
        all_values['nominal_reference'].append((v['logpdf'], v['mu'], v['class_distance'], True))
        lk, _ = kernel_loglike(response, obs, g, calibration_mag=cal[row])
        v = marginalize_native(response, g, lk, theta[int(fields[row])], beta, weights)
        good = v['joint_delta'] < 1e-8 and abs(v['log_normalization']) < 1e-8 and v['marginal_half_delta'] < .005
        all_values['old_target'].append((v['full_logpdf'], v['posterior'].quantiles(),
                                          v['class_distance'], good))
    return all_values


def stack_view(rows, grids):
    p = {}
    levels = []
    for j, prefix in enumerate(('coarse_', '')):
        lp, q, jcd, status = map(np.asarray, zip(*(r[j] for r in rows)))
        p.update({prefix+'logpdf': lp, prefix+'mu_quantiles': q,
                  prefix+'class_distance': jcd, prefix+'mu_grid': grids[j]})
        levels.append((q, jcd, status))
    (q0, p0, a), (q1, p1, b) = levels
    q_delta = np.max(abs(q1-q0), axis=1)
    probability_delta = np.max(abs(p1-p0), axis=(1, 2))
    p.update(numerical_pass=a & b & (q_delta < .03) & (probability_delta < .003),
             quantile_delta=q_delta, probability_delta=probability_delta)
    return p


def old_population(model, photo, chain, fields):
    field_ids, in_old = geographic_fields(photo['RA'], photo['DEC'], photo['cell'], fields)
    if in_old.any():
        raise ValueError('Prospective objects overlap a historical target footprint')
    original = np.array(sorted(fields))
    shape = (len(original), len(model.weights), 5)
    fit = FieldPopulation(np.ones(shape), np.ones(shape), model, original)
    decoded = [fit.decode(z) for z in chain['samples']]
    beta = np.array([v[1] for v in decoded])
    logits = np.array([v[3] for v in decoded])
    theta = {}
    for field in np.unique(field_ids):
        rng = np.random.default_rng(np.random.SeedSequence([316101, 12, int(field)]))
        local = logits+.35*rng.normal(size=logits.shape)
        theta[int(field)] = softmax(np.c_[local, np.zeros(len(local))], axis=1).reshape(-1, 3, 5)
    return field_ids, theta, beta


def main():
    global _STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 33), default=16)
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument('--common', action='store_true', help='Use the outcome-closed common-system census')
    cohort.add_argument('--minesweeper', action='store_true', help='Use the closed DR20 field census')
    args = parser.parse_args()
    require_compute()
    root = ROOT/'data/v3_2'
    native_name, prefix, role = ('common_scale_inventory', 'common_scale', COMMON_ROLE) if args.common else (
        ('minesweeper_native', 'minesweeper', MINESWEEPER_ROLE) if args.minesweeper else
        ('reference_native', 'reference', ROLE))
    native = root/native_name/'complete.json'
    census = json.loads(native.read_text())
    closed = (census.get('scale_values_exported') is False and census.get('predictions_read') is False
              if args.common else census.get('reference_values_read') is False)
    if not closed or census.get('all_admitted_PSF') is not True:
        raise ValueError('Require a photometry-only admission')
    photo_path = verified(native.parent/'photometry.npz', census['outputs']['photometry.npz'])
    photo = load(photo_path)
    require_native(photo)
    if args.minesweeper and len(photo['identity']) != census['admitted']:
        raise ValueError('DR20 photometry differs from the native census')
    if args.common and (len(photo['identity']) != census['admitted_members']
                        or len(np.unique(photo['scale_system'])) != census['admitted_systems']):
        raise ValueError('Common-system photometry differs from the native census')
    old, registered = ROOT/'data/v3_0', ROOT/'manifests/v3_0'
    sed_receipt = registered/'sed_fit.json'
    qrf_receipt = registered/'controls/complete.json'
    v31_receipt = ROOT/'manifests/v3_1/validation_old_target/field/complete.json'
    recipe = json.loads(v31_receipt.read_text())['inputs']

    def frozen(name):
        return verified(ROOT/name, recipe[name])

    nominal_path = verified(old/'sed_fit.joblib', json.loads(sed_receipt.read_text())['sha256'])
    qrf_path = verified(old/'controls/qrf_grz.joblib',
                        json.loads(qrf_receipt.read_text())['outputs']['qrf_grz.joblib'])
    response_path = frozen('data/v3_0/uncertainty_refit/draw_12/model.joblib')
    shared_path = frozen('data/v3_0/uncertainty/shared_draws.npz')
    chain_path = frozen('data/v3_0/native_responses/refit_12_s4096/chain_correction.npz')
    app_path = frozen('data/v3_0/cohorts/application.npz')
    prep_path = frozen('data/baseline_v0/prepared.json')
    qrf = joblib.load(qrf_path)
    admission = candidate_admission(photo, qrf)
    nominal, response = [inference_model(joblib.load(p)) for p in (nominal_path, response_path)]
    for model in (nominal, response):
        if not model.converged or np.intersect1d(model.training_identity, photo['identity']).size:
            raise ValueError('Frozen SED model failed fit-identity qualification')
    with np.load(app_path) as app:
        original = np.unique(app['field'])
        if np.intersect1d(photo['identity'], app['identity']).size:
            raise ValueError('Reference identities overlap frozen application fit')
    configs, bounds = [], {}
    for f in json.loads(prep_path.read_text())['fields']:
        if f['index'] in original:
            p = frozen(f['config'])
            configs.append(p)
            bounds[f['index']] = json.loads(p.read_text())['bounds_deg']
    if set(bounds) != set(original):
        raise ValueError('Missing historical field definitions')
    cal, _ = calibration_vectors(photo, load(shared_path), 12)
    chain = load(chain_path)
    fields, theta, beta = old_population(response, photo, chain, bounds)
    grids = [np.unique(np.r_[np.linspace(-20., 50., n), DISTANCE_MU_BOUNDARIES])
             for n in (3501, 7001)]
    _STATE = photo, nominal, response, cal, grids, fields, theta, beta, chain['weights']
    destination = root/f'{prefix}_controls'
    if destination.exists():
        raise FileExistsError(destination)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as workers:
        rows = list(workers.map(continuous_views, range(len(photo['identity']))))
    views = {name: stack_view([r[name] for r in rows], grids) for name in rows[0]}
    views['qrf'] = qrf_prediction(qrf, photo)
    destination.mkdir(parents=True, exist_ok=False)
    admission_path = destination/'candidate_admission.npz'
    atomic_npz(admission_path, ls_id=photo['ls_id'], **admission)
    admission_export = dict(payload=str(admission_path.relative_to(ROOT)), sha256=sha(admission_path),
                            admitted=int(admission['candidate'].sum()))
    exports = {}
    for name, data in views.items():
        p = destination/f'{name}.npz'
        atomic_npz(p, **{k: photo[k] for k in IDENTITY}, **data)
        exports[name] = dict(payload=str(p.relative_to(ROOT)), sha256=sha(p),
                             passed=int(data['numerical_pass'].sum()))
    shared = destination/'old_target_shared.npz'
    atomic_npz(shared, draw_ids=chain['draw_ids'], weights=chain['weights'], beta=beta,
               field_ids=fields, field=np.array(sorted(theta)),
               theta=np.stack([theta[key] for key in sorted(theta)], axis=1),
               **{k: photo[k] for k in IDENTITY})
    sources = configs+[native, photo_path, sed_receipt, qrf_receipt, v31_receipt, nominal_path,
                       qrf_path, response_path, shared_path, chain_path, app_path, prep_path,
                       Path(__file__).resolve()]
    sources += [ROOT/p for p in ('scripts/v3_1_luminosity_guard.py', 'scripts/v3_1_validation_predict.py',
                'src/photdis/v3_1/admission.py', 'src/photdis/v3_1/shared_prior.py',
                'src/photdis/v3/controls.py', 'src/photdis/baseline_v0.py',
                'src/photdis/baseline_v1.py', 'src/photdis/v3/inference.py',
                'src/photdis/v3/systematics.py', 'src/photdis/v3/hierarchy.py')]
    save_json(destination/'complete.json', dict(cohort=role,
              n=len(photo['identity']), all_PSF=True,
              references_read=False, pilot=False, photometry_sha256=sha(photo_path), views=exports,
              candidate_admission=admission_export,
              inputs={str(p.relative_to(ROOT)): sha(p) for p in sources},
              outputs={p.name: sha(p) for p in destination.glob('*.npz')}, job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Make closed validation forecasts with the registered observable working points."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp

from photdis.stage1 import require_compute
from photdis.v3.controls import apply_temperature, optical_magnitudes, probe_features
from photdis.v3.model import inference_model
from photdis.v3_3 import class_log_likelihood, frozen_qrf, reference_log_curves
from v2_1_near import ROOT, atomic_npz, load, save_json, sha
from v3_0_response import observation
from v3_3_catalog import FREEZE, joint_summary, read_frozen
from v3_3_sed import BASE, MANIFESTS, verified
from v3_4_near_development import DISTANCE_EDGES, MU, OUT as COMPARISON
from v3_4_near_validation_inputs import CONFIG, OUT as INPUTS, receipt_payload

OUT = INPUTS.parent/'forecast'
_STATE = None


def prepare():
    entries = read_frozen()['models']
    paths = {k: verified(ROOT/entries[k]['path'], entries[k]['sha256']) for k in ('nominal', 'qrf')}
    control = MANIFESTS/'controls/complete.json'
    probe_path = verified(BASE/'controls/probes.joblib', json.loads(control.read_text())['outputs']['probes.joblib'])
    probe = joblib.load(probe_path)['grz']
    np.testing.assert_array_equal(probe['model'].classes_, [0, 1, 2])
    return {k: joblib.load(p) for k, p in paths.items()}, probe, [FREEZE, control, probe_path, *paths.values()]


def nominal_row(i):
    photo, model, prior, retain_curve = _STATE
    ll = class_log_likelihood(model, observation(photo, i), MU)
    summary = joint_summary(MU, ll, prior)
    curve = logsumexp(reference_log_curves(ll, MU, prior), axis=0).astype('f4') if retain_curve else None
    return summary['qnear'], summary['numerical_flag'], curve


def qrf_bins(edges, mass):
    bounds = np.r_[-np.inf, 10+5*np.log10(DISTANCE_EDGES[1:])]
    cdf = np.empty((len(mass), len(bounds)))
    for j, b in enumerate(bounds):
        cdf[:, j] = np.sum(np.clip((b-edges[:, :-1])/np.diff(edges), 0, 1)*mass, axis=1)
    return np.diff(cdf, axis=1)


def forecast(photo, cfg, models, probe, *, retain_curve):
    global _STATE
    if not (photo['TYPE'] == 'PSF').all():
        raise ValueError('Forecasts require verified native Legacy PSFs')
    dwarf = apply_temperature(probe['model'].predict_proba(probe_features(photo, 'grz')), probe['temperature'])[:, 0]
    mag, err = optical_magnitudes(photo)
    qrf = frozen_qrf(models['qrf'], mag, err, photo['identity'])
    nominal = models['nominal']
    _STATE = photo, inference_model(nominal), nominal.weights @ nominal.classes, retain_curve
    with ProcessPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '16')),
                             mp_context=mp.get_context('fork')) as pool:
        nominal_q, flags, curves = zip(*pool.map(nominal_row, range(len(photo['identity'])), chunksize=12))
    _STATE = None
    nominal_q = np.asarray(nominal_q)
    bins = qrf_bins(qrf['mu_edges'], qrf['mass'])
    np.testing.assert_allclose(bins.sum(axis=1), 1., atol=1e-12, rtol=0)
    cuts = {k: qrf['supported'] & (dwarf >= cfg[k]['minimum_dwarf_score'])
            & (nominal_q >= cfg[k]['minimum_near_probability']) for k in ('primary', 'strict')}
    if np.any(cuts['strict'] & ~cuts['primary']):
        raise ValueError('Strict sample lies outside the primary selection')
    result = dict(identity=photo['identity'], ls_id=photo['ls_id'], TYPE=photo['TYPE'],
        qrf_supported=qrf['supported'], dwarf_selection_score=dwarf, nominal_qnear=nominal_q,
        nominal_numerical_flag=np.asarray(flags), qrf_quantiles_kpc=qrf['distance'],
        qrf_qnear=bins[:, :3].sum(axis=1), qrf_bins=bins, qrf_mu_edges=qrf['mu_edges'],
        qrf_mu_mass=qrf['mass'], **cuts)
    if retain_curve:
        result.update(nominal_mu=MU, nominal_logp=np.asarray(curves))
    return result


def check_development_replay(application, predicted, inputs):
    roster = receipt_payload(COMPARISON, 'roster.npz', inputs)
    qrf = receipt_payload(COMPARISON, 'qrf.npz', inputs)
    nominal = receipt_payload(COMPARISON, 'nominal.npz', inputs)
    app = np.flatnonzero(roster['application_sample'])
    order = np.argsort(roster['identity'][app])
    app = app[order]
    new = np.argsort(application['identity'])
    target = application['prediction_row'][new]
    for key in ('identity', 'ls_id', 'TYPE'):
        np.testing.assert_array_equal(predicted[key][target], roster[key][app])
    np.testing.assert_array_equal(predicted['primary'][target], qrf['near_dwarf_comparison'][app])
    np.testing.assert_array_equal(predicted['strict'][target], qrf['strict_near_dwarf_comparison'][app])
    np.testing.assert_allclose(predicted['qrf_qnear'][target], qrf['qnear'][app], atol=1e-12, rtol=0)
    np.testing.assert_allclose(predicted['nominal_qnear'][target], nominal['qnear'][app], atol=1e-12, rtol=0)


def main():
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (CONFIG, Path(__file__))}
    photo = receipt_payload(INPUTS, 'photometry.npz', inputs)
    app = receipt_payload(INPUTS, 'application.npz', inputs)
    fresh = receipt_payload(INPUTS, 'fresh_gaia.npz', inputs)
    models, probe, sources = prepare()
    for name, model in models.items():
        identities = np.r_[fresh['identity'], fresh['source_id']]
        if np.intersect1d(model.training_identity, identities).size:
            raise ValueError(f'Training identities in new Gaia evidence for {name}')
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in sources})
    predicted = forecast(photo, cfg, models, probe, retain_curve=True)
    check_development_replay(app, predicted, inputs)
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT/'forecasts.npz'
    atomic_npz(path, **predicted)
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), all_PSF=True,
        new_outcomes_read=False, n=len(photo['identity']), replay_n=len(app['identity']),
        selected={k:{'application':int(predicted[k][app['prediction_row']].sum()),
                     'fresh_gaia':int(predicted[k][fresh['prediction_row']].sum())} for k in ('primary', 'strict')},
        inputs=inputs, outputs={path.name:sha(path)}))


if __name__ == '__main__':
    main()

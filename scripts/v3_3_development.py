#!/usr/bin/env python3
"""Evaluate the first partitioned SED and frozen baselines on opened identities."""
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp

from photdis.audit import rvs_metallicity
from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis.v3.data import stable_rank
from photdis.v3.likelihood import normalize_joint
from photdis.v3.metrics import (distance_summary, histogram_parallax, histogram_reference_score,
                                normal_reference_score, observed_parallax)
from photdis.v3.model import inference_model
from photdis.v3_1.validation_report import paired_field_statistic
from photdis.v3_3 import class_log_likelihood, reference_log_curves
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
import v3_0_evaluate as historical
from v3_0_response import observation
from v3_3_sed import BASE, MANIFESTS, OUT as FIT, verified

OUT = ROOT/'data/v3_3/development/initial'
MU = np.linspace(-5., 30., 1401)
VIEWS = ('nominal', 'candidate', 'qrf')
_STATE = None


def extract_row(index):
    data, reference, models, prior, qrf_edges, qrf_prediction = _STATE
    label = data['class_label'][index]
    o = observation(data, index)
    center, error, valid, fraction = (a[index] for a in reference)
    source = data['primary_channel'][index]
    broad = (np.hypot(data[('specdis', 'spdist')[source-1]+'_sigma_mu'][index], 2.)
             if source in (1, 2) else 2.)
    parallax, parallax_error = (data[k][index] for k in ('p_corr', 'gaia_parallax_error'))
    gaia = data['parallax_eligible'][index]
    encoded = []
    for model in models:
        likelihood = class_log_likelihood(model, o, MU)
        curve = reference_log_curves(likelihood, MU, prior)
        p = normalize_joint(MU, curve)
        coarse = normalize_joint(MU[::2], curve[:, ::2])
        class_p = p.component_mass()
        qnear = p.cdf(15.).sum()
        quantiles = p.quantiles()
        probability_delta = max(abs(class_p-coarse.component_mass()).max(), abs(qnear-coarse.cdf(15.).sum()))
        modulus_delta = abs(quantiles-coarse.quantiles()).max()
        marginal = logsumexp(curve, axis=0)
        oracle_q = np.full(5, np.nan)
        oracle_score = np.nan
        oracle_qnear = np.nan
        oracle_gaia = np.nan
        if label >= 0:
            selected = curve[label:label+1]
            conditional = normalize_joint(MU, selected)
            oracle_q = conditional.quantiles()
            coarse_conditional = normalize_joint(MU[::2], selected[:, ::2])
            modulus_delta = max(modulus_delta, abs(oracle_q-coarse_conditional.quantiles()).max())
            oracle_qnear = conditional.cdf(15.).sum()
            probability_delta = max(probability_delta, abs(oracle_qnear-coarse_conditional.cdf(15.).sum()))
            if valid:
                oracle_score = normal_reference_score(MU, selected[0], center, error,
                                                      broad_fraction=fraction, broad_error=broad)
            if gaia:
                oracle_gaia = observed_parallax(MU, selected[0], parallax, parallax_error)['log_score']
        gaia_score = observed_parallax(MU, marginal, parallax, parallax_error)['log_score'] if gaia else np.nan
        score = normal_reference_score(MU, marginal, center, error,
                                       broad_fraction=fraction, broad_error=broad) if valid else np.nan
        encoded.append(dict(quantiles=quantiles, qnear=qnear, class_probability=class_p, oracle_quantiles=oracle_q,
            oracle_qnear=oracle_qnear, reference_score=score, gaia_score=gaia_score,
            oracle_reference_score=oracle_score, oracle_gaia_score=oracle_gaia,
            probability_delta=probability_delta, modulus_delta=modulus_delta))

    edges = qrf_edges[index]
    mass = qrf_prediction['pdf'][index, ::-1].astype(float)
    mass /= mass.sum()
    qrf_qnear = np.sum(mass*np.clip((15.-edges[:-1])/np.diff(edges), 0, 1))
    gaia_score = histogram_parallax(edges, mass, parallax, parallax_error, order=32)['log_score'] if gaia else np.nan
    score = histogram_reference_score(edges, mass, center, error, fraction, broad) if valid else np.nan
    encoded.append(dict(quantiles=qrf_prediction['mu_quantiles'][index], qnear=qrf_qnear,
                       reference_score=score, gaia_score=gaia_score, supported=qrf_prediction['supported'][index]))
    return encoded


def summarize(outputs, data, center, feh):
    groups = {'all': np.ones(len(center), bool), 'metal_poor': feh < -1.5}
    groups.update({f'class_{c}': data['class_label'] == c for c in (0, 1, 2)})
    groups['metal_poor_giant'] = groups['class_1'] & groups['metal_poor']
    distance = 10**((center-10.)/5.)
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, np.inf)):
        use = (distance >= lo) & (distance < hi)
        groups[f'distance_{lo}_{hi}'] = use
        groups[f'giant_distance_{lo}_{hi}'] = use & groups['class_1']
    result, contrasts = [], []
    for group, use in groups.items():
        for view, values in outputs.items():
            for condition, key in (('marginal', 'quantiles'), ('known_class', 'oracle_quantiles')):
                if key in values:
                    result.append(dict(group=group, view=view, condition=condition,
                                       **distance_summary(values[key][use], center[use])))
        for baseline in ('nominal', 'qrf'):
            for metric in ('gaia_score', 'reference_score', 'oracle_gaia_score', 'oracle_reference_score'):
                if metric in outputs[baseline]:
                    contrast = paired_field_statistic(outputs['candidate'][metric][use], outputs[baseline][metric][use],
                                                      data['cell'][use])
                    contrasts.append(dict(group=group, baseline=baseline, metric=metric, **contrast))
    return result, contrasts


def main():
    require_compute()
    cohorts = json.loads((MANIFESTS/'cohorts.json').read_text())['outputs']
    paths = [verified(BASE/path, cohorts[path]) for path in ('cohorts/development.npz', 'evaluation/development.npz')]
    photo, evidence = map(load, paths)
    np.testing.assert_array_equal(photo['identity'], evidence['identity'])
    strict = (photo['TYPE'] == 'PSF') & (evidence['TYPE'] == 'PSF')
    if not (evidence['role'] == 4).all():
        raise ValueError('Expected opened development roles only')
    eligible = strict & evidence['reference_stellar_eligible']
    selected = []
    for label, maximum in ((-1, 512), (0, 1024), (1, None), (2, 1024)):
        rows = np.flatnonzero(eligible & (evidence['class_label'] == label))
        if maximum is not None:
            rows = rows[np.argsort(stable_rank(photo['identity'][rows], 330527), kind='stable')[:maximum]]
        selected.extend(rows)
    rows = np.sort(selected)
    data = {key: val[rows] for key, val in (photo | evidence).items()}
    nominal_marker = json.loads((MANIFESTS/'sed_fit.json').read_text())
    fit_marker = json.loads((FIT/'complete.json').read_text())
    nominal_path = verified(BASE/'sed_fit.joblib', nominal_marker['sha256'])
    candidate_path = verified(FIT/'model.joblib', fit_marker['sha256'])
    qrf_name = 'qrf_grz.joblib'
    qrf_path = verified(BASE/'controls'/qrf_name,
                        json.loads((MANIFESTS/'controls/complete.json').read_text())['outputs'][qrf_name])
    nominal, candidate, qrf = map(joblib.load, (nominal_path, candidate_path, qrf_path))
    for model in (nominal, candidate, qrf):
        if np.intersect1d(model.training_identity, data['identity']).size:
            raise ValueError('Training identity in development sample')
    prior = nominal.weights @ nominal.classes
    cal_path = verified(BASE/'calibration.npz', json.loads((MANIFESTS/'calibration.json').read_text())['sha256'])
    historical._CAL = load(cal_path)
    reference = historical.evidence_views(data)['primary']
    mags, errors = optical_magnitudes(data)
    qrf_prediction = qrf.predict(mags, errors, data['identity'])
    qrf_prediction['mu_quantiles'] = mags[:, :1] - qrf_prediction['q_Meffective'][:, ::-1]
    qrf_edges = mags[:, :1] - qrf.edges[None, ::-1]
    global _STATE
    _STATE = data, reference, list(map(inference_model, (nominal, candidate))), prior, qrf_edges, qrf_prediction
    with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('fork')) as pool:
        results = list(pool.map(extract_row, range(len(rows)), chunksize=8))
    outputs = {view: {key: np.asarray([result[i][key] for result in results]) for key in results[0][i]}
               for i, view in enumerate(VIEWS)}
    feh, feh_valid = rvs_metallicity(data['rvs_feh'], data['rvs_teff'], data['rvs_logg'])
    feh = np.where(feh_valid & data['rvs_parameter_quality'], feh, np.nan)
    summaries, contrasts = summarize(outputs, data, reference[0], feh)
    OUT.mkdir(parents=True, exist_ok=False)
    for view, values in outputs.items():
        atomic_npz(OUT/f'{view}.npz', identity=data['identity'], TYPE=data['TYPE'], reference_mu=reference[0],
                    reference_class=data['class_label'], corrected_feh=feh, cell=data['cell'], **values)
    csv_rows(OUT/'distance.csv', summaries)
    csv_rows(OUT/'paired_scores.csv', contrasts)
    sources = [*paths, nominal_path, candidate_path, qrf_path, cal_path, Path(__file__).resolve(),
               ROOT/'src/photdis/v3_3.py', ROOT/'scripts/v3_0_evaluate.py']
    for view in ('nominal', 'candidate'):
        np.testing.assert_allclose(outputs[view]['class_probability'].sum(axis=1), 1., rtol=0, atol=1e-11)
    save_json(OUT/'complete.json', dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in sources},
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir())}, all_PSF=True, n=len(rows),
        missing_TYPE=int(((photo['TYPE'] == '') | (evidence['TYPE'] == '')).sum()),
        non_PSF=int((~strict & (photo['TYPE'] != '') & (evidence['TYPE'] != '')).sum()),
        candidate_fit_converged=fit_marker['converged'],
        grid_flags={view: int(((outputs[view]['probability_delta'] > .003) |
                              (outputs[view]['modulus_delta'] > .03)).sum()) for view in ('nominal', 'candidate')},
        job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    main()

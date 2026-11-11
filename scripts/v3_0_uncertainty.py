#!/usr/bin/env python3
"""Shared one-step finite-reference/source-calibration SED response ensemble."""
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, atomic_npz, save_json
from photdis.stage1 import require_compute
from photdis.v3.uncertainty import (reference_quadrature_components, reference_logratio,
    shared_case_weights, one_step_ensemble)

D = ROOT/'data/v3_0'


def main():
    require_compute(); start = time.monotonic()
    destination = D/'uncertainty'; destination.mkdir(exist_ok=True)
    paths = [D/'sed_fit.joblib', D/'sed_cloud.npz', D/'calibration.npz',
             D/'references.npz', ROOT/'src/photdis/v3/uncertainty.py',
             ROOT/'scripts/v3_0_uncertainty.py']
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    freeze = destination/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze, inputs)
    marker = destination/'complete.json'
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == inputs
        for name, digest in record['outputs'].items(): assert sha(destination/name) == digest
        return
    model = joblib.load(D/'sed_fit.joblib')
    if not model.converged: raise RuntimeError('Full SED fit not converged; diagnose before ensemble')
    cloud = load(D/'sed_cloud.npz'); t = load(D/'references.npz'); cal = load(D/'calibration.npz')
    rows = cloud['rows']; channel = cloud['channel']; n = len(rows)
    assert np.array_equal(model.training_identity, cloud['identity'])
    assert (t['TYPE'][rows] == 'PSF').all() and (t['role'][rows] == 1).all()
    reported_mu = np.zeros(n); reported_sigma = np.zeros(n)
    for c, source in [(1, 'specdis'), (2, 'spdist')]:
        use = channel == c
        reported_mu[use] = 10+5*np.log10(t[source+'_distance'][rows][use])
        reported_sigma[use] = t[source+'_sigma_mu'][rows][use]
    broad = reference_quadrature_components(cloud['reference_mu'], channel,
        reported_mu, reported_sigma, cal['map_parameters'])
    ratio, norm = reference_logratio(cloud['reference_mu'], channel, reported_mu,
        reported_sigma, cal['map_parameters'], cal['parameters'], cloud['reference_weight'], broad)
    # Numerical quadrature diagnostics are exported, never relabeled uncertainty.
    normalization_audit = dict(maximum=float(abs(norm).max()),
        q99=float(np.quantile(abs(norm), .99)), above_0p01=int((abs(norm) > .01).sum()))
    if normalization_audit['q99'] > .01:
        raise RuntimeError('Reference importance quadrature underresolved: '+str(normalization_audit))
    weights, cells, sky = shared_case_weights(cloud['identity'], t['cell'][rows], draws=len(ratio))
    fields = np.r_[np.arange(14), [100, 101]]
    del t
    print('SHARED RESPONSE', len(ratio), 'draws', n, 'sources', normalization_audit, flush=True)
    ensemble, evidence = one_step_ensemble(model, cloud, ratio, weights, workers=16)
    joblib.dump(ensemble, destination/'ensemble.tmp.joblib')
    (destination/'ensemble.tmp.joblib').replace(destination/'ensemble.joblib')
    rng = np.random.default_rng(301802)
    global_zero_point = rng.normal(size=(len(ensemble), 6))*np.array([.005, .005, .005, .005, .01, .01])
    field_dust = rng.normal(size=(len(ensemble), len(fields)))*.003
    atomic_npz(destination/'shared_draws.npz', draw_id=np.arange(len(ensemble), dtype='i8'),
        source_calibration=cal['parameters'], calibration_draw_id=cal['draw_ids'],
        training_identity=cloud['identity'], TYPE=cloud['TYPE'], training_sky_cells=cells,
        training_sky_weights=sky, field_ids=fields, field_dust_offset=field_dust,
        global_band_zero_point_mag=global_zero_point,
        reference_quadrature_log_normalization=norm,
        mean_reference_log_evidence=evidence.mean(axis=1))
    outputs = {p.name: sha(p) for p in destination.iterdir() if p.is_file() and p.name != marker.name}
    save_json(marker, dict(inputs=inputs, outputs=outputs, n=n, draws=len(ensemble),
        method='one EM influence update, common sky/identity Bayesian bootstrap plus all source-calibration parameters',
        limits='not fully refitted mixture posterior; recovery/coverage and full-update sensitivity required',
        system_prior='global band zeropoint sigma 0.005 optical/0.01 WISE mag; field E(B-V) sigma 0.003, declared sensitivity priors not measured calibration errors',
        reference_quadrature=normalization_audit, elapsed_seconds=time.monotonic()-start,
        job=os.getenv('SLURM_JOB_ID')))
    print('SHARED RESPONSE COMPLETE', len(ensemble), time.monotonic()-start, flush=True)


if __name__ == '__main__': main()

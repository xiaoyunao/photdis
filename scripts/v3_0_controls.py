#!/usr/bin/env python3
"""Fixed QRF-grz and class-information controls on exactly the SED identities."""
import json
import os
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from v2_1_near import ROOT, load, save_json, sha, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.baseline_v0 import HistogramQRF
from photdis.v3.controls import (optical_magnitudes, probe_features,
    calibrate_temperature, apply_temperature, probability_metrics, reliability_rows)

D = ROOT/'data/v3_0'


def main():
    require_compute(); start = time.monotonic()
    destination = D/'controls'; destination.mkdir(exist_ok=True)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [D/'references.npz', D/'sed_cloud.npz',
        ROOT/'src/photdis/v3/controls.py', ROOT/'src/photdis/baseline_v0.py',
        ROOT/'scripts/v3_0_controls.py']}
    freeze = destination/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs, 'Explicit control migration required'
    else: save_json(freeze, inputs)
    marker = destination/'complete.json'
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == inputs
        for name, digest in record['outputs'].items(): assert sha(destination/name) == digest
        return
    t = load(D/'references.npz')
    with np.load(D/'sed_cloud.npz') as z:
        train = z['rows']; identity = z['identity']; labels = z['labels']
        ref_mu = (z['reference_weight']*z['reference_mu']).sum(axis=1)
    assert np.array_equal(t['identity'][train], identity)
    assert (t['TYPE'][train] == 'PSF').all() and (t['role'][train] == 1).all()
    mag, _ = optical_magnitudes(t)
    # QRF point-label control uses the SAME primary evidence/calibration as SED.
    # It deliberately lacks SED deconvolution; this is the method contrast.
    y = mag[train, 0]-ref_mu
    qrf_path = destination/'qrf_grz.joblib'
    if not qrf_path.exists():
        qrf = HistogramQRF(2); qrf.forest.n_jobs=16
        qrf.fit(mag[train], y, labels); qrf.forest.n_jobs=1
        qrf.training_identity = identity.copy()
        joblib.dump(qrf, qrf_path.with_suffix('.tmp')); qrf_path.with_suffix('.tmp').replace(qrf_path)
    atomic_npz(destination/'training.npz', identity=identity, TYPE=t['TYPE'][train],
               rows=train, Mg=y, primary_mu=ref_mu, class_label=labels)
    train_class = train[labels >= 0]
    hyper = np.flatnonzero((t['role'] == 3) & (t['class_label'] >= 0))
    dev = np.flatnonzero((t['role'] == 4) & (t['class_label'] >= 0))
    assert not np.intersect1d(identity, t['identity'][hyper]).size
    assert not np.intersect1d(identity, t['identity'][dev]).size
    metrics = []; reliability = []; predictions = dict(identity=t['identity'][dev],
        TYPE=t['TYPE'][dev], rows=dev, class_label=t['class_label'][dev])
    models = {}
    for view in ('no_color', 'grz', 'griz', 'multiband'):
        x = probe_features(t, view)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05,
            max_leaf_nodes=15, l2_regularization=1., early_stopping=False,
            random_state=301711)
        clf.fit(x[train_class], t['class_label'][train_class])
        assert np.array_equal(clf.classes_, np.arange(3))
        temperature = calibrate_temperature(clf.predict_proba(x[hyper]), t['class_label'][hyper])
        raw = clf.predict_proba(x[dev]); p = apply_temperature(raw, temperature)
        predictions[view] = p; predictions[view+'_raw'] = raw
        models[view] = dict(model=clf, temperature=temperature)
        groups = {'all': np.ones(len(dev), bool),
                  'all_optional_measured': t['band_mask'][dev][:, [2, 4, 5]].all(axis=1),
                  'weak_W1': t['band_state'][dev, 4] == 3,
                  'no_W1': ~t['band_mask'][dev, 4]}
        groups.update({f'class_{c}': t['class_label'][dev] == c for c in range(3)})
        observed_r = mag[dev, 1]-2.5*np.log10(t['transmission'][dev, 1])
        for low, high in [(0, 18), (18, 20), (20, 22), (22, 99)]:
            groups[f'r_{low}_{high}'] = (observed_r >= low) & (observed_r < high)
        for group, use in groups.items():
            for version, probability in [('raw', raw), ('calibrated', p)]:
                metrics.append(dict(view=view, group=group, version=version,
                    temperature=temperature, **probability_metrics(probability[use], t['class_label'][dev][use])))
        reliability.extend(dict(view=view, **r) for r in reliability_rows(p, t['class_label'][dev]))
        print('FIXED PROBE', view, 'temperature', temperature, flush=True)
    joblib.dump(models, destination/'probes.joblib')
    atomic_npz(destination/'probe_development.npz', **predictions)
    csv_rows(destination/'probe_metrics.csv', metrics)
    csv_rows(destination/'probe_reliability.csv', reliability)
    outputs = {p.name: sha(p) for p in destination.iterdir() if p.is_file() and p.name != marker.name}
    save_json(marker, dict(inputs=inputs, outputs=outputs, train_sources=len(train),
        classifier_training_sources=len(train_class), missing_class_distance_sources=int((labels < 0).sum()),
        temperature_calibration_sources=len(hyper), development_sources=len(dev),
        qrf_evidence='same primary-channel posterior mean mu, point-label QRF; fixed historical hyperparameters',
        probe='one fixed HistGradientBoosting family; no-color/grz/griz/multiband, common train and evaluation identities',
        class_weighting='unweighted empirical reference composition, class-specific metrics separate',
        prospective_outcomes_opened=False, elapsed_seconds=time.monotonic()-start,
        job=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Broad grz-only same-label probe, including objects with no usable WISE."""
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT, DEST, TABLES, load, subset, csv_rows, sha, save_json, atomic_npz
from v2_1_information import groups, design
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.probability_v2_1 import psf_admission
from photdis.baseline_v1_2 import DirectClassifier, balanced_weights
from photdis.information_v2_1 import leaf_class_control, fit_calibration, calibrated, class_score


def main():
    require_compute()
    model_path = ROOT/'data/baseline_v1_4/experiments/expanded_A_distance.joblib'
    model = joblib.load(model_path)
    train = load(DEST/'frozen_training_audit.npz')
    eligible = np.zeros(len(train['identity']), bool)
    parts = []
    keys = ['identity', 'ls_id', 'split', 'cell', 'class_label', 'class_eligible', 'class_train_eligible',
            'photometric_eligible', 'identity_conflict', 'nonstellar_risk', 'r', 'optical_error',
            'B', 'C', 'g0', 'r0', 'z0', 'sigma_g', 'sigma_r', 'sigma_z']
    for version in ('v1_3', 'v1_4'):
        path = ROOT/f'data/baseline_{version}/references.npz'
        with np.load(path) as z:
            j = unique_join(train['identity'], z['identity'])
            matched = j >= 0
            # Old/new catalogs can contain the same historical identity; original
            # old members take precedence, and new admissible train is disjoint.
            if version == 'v1_3':
                old_member = np.isin(train['identity'], load(ROOT/'data/baseline_v1_4/experiments/expanded_A_legacy_training.npz')['distance_train_identity'])
                matched &= old_member
            else:
                matched &= ~old_member
            eligible[matched] = z['class_train_eligible'][j[matched]]
            if version == 'v1_4':
                t = {k: z[k] for k in keys}
    good, audit = psf_admission(train['TYPE'])
    good &= eligible & np.isin(train['class_label'], [0, 1, 2])
    tr = subset(train, good)
    native = load(ROOT/'data/v2_0/native_report.npz')
    j = unique_join(t['ls_id'], native['ls_id'])
    t['TYPE'] = np.full(len(j), '', dtype='U4')
    t['TYPE'][j >= 0] = native['TYPE'][j[j >= 0]]
    psf, evaudit = psf_admission(t['TYPE'])
    use = (psf & np.isin(t['split'], [1, 2]) & t['class_eligible'] & t['photometric_eligible']
           & ~t['identity_conflict'] & ~t['nonstellar_risk'])
    t = subset(t, use)
    assert not np.intersect1d(tr['identity'], t['identity']).size
    tm, _ = design(tr, ('g', 'r', 'z'))
    em, _ = design(t, ('g', 'r', 'z'))
    labels, y = tr['class_label'].astype(int), t['class_label'].astype(int)
    prior = np.bincount(labels, minlength=3)/len(labels)
    cfg = json.loads((ROOT/'configs/baseline_v1_2.json').read_text())
    clf = DirectClassifier(2, 'forest', cfg).fit(tm, labels, balanced_weights(labels))
    joblib.dump(clf, DEST/'probe_broad_grz.joblib')
    same, empty = leaf_class_control(model, tm, labels, em, prior)
    # Original unknown channel is retained separately and never interpreted as
    # a physical fourth population. Known-class conditioning is explicit.
    original = model.distribution(model.feature_matrix(em))[1]
    known_mass = original[:, :3].sum(axis=1)
    original_known = np.divide(original[:, :3], known_mass[:, None],
                               out=np.broadcast_to(prior, (len(y), 3)).copy(), where=known_mass[:, None] > 0)
    raw = dict(no_color=np.broadcast_to(prior, (len(y), 3)).copy(),
               frozen_partitions_same_labels=same, fixed_forest=clf.predict_proba(em),
               historical_QRF_known_conditional=original_known)
    probability, calibration = {}, {}
    for method, p in raw.items():
        cal = t['split'] == 1
        calibration[method] = fit_calibration(p[cal], y[cal])
        probability[method] = calibrated(p, calibration[method])
    scores = []
    masks = groups(t)
    masks.update(no_W1=~t['B'], W1_only=t['B'] & ~t['C'], W1W2=t['C'])
    masks['r18_20:no_W1'] = (t['r'] >= 18) & (t['r'] < 20) & ~t['B']
    for split, role in [(1, 'cal_fit'), (2, 'development_test')]:
        for group, mask in masks.items():
            use = mask & (t['split'] == split)
            for method, p in probability.items():
                scores.append(dict(split=role, group=group, method=method,
                    **class_score(p[use], y[use], probability['no_color'][use]),
                    historical_unknown_mass=float(np.mean(original[use, 3])) if use.any() else None))
    csv_rows(TABLES/'optical_probe.csv', scores)
    atomic_npz(DEST/'optical_probe.npz', identity=t['identity'], TYPE=t['TYPE'], class_label=y,
               split=t['split'], cell=t['cell'], train_identity=tr['identity'],
               **probability)
    save_json(DEST/'optical_probe_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'),
        model_sha256=sha(model_path), training_morphology=audit, evaluation_morphology=evaudit,
        train_n=len(labels), eval_n=len(y), calibration=calibration,
        interpretation='Primary broad optical same-label probability probe; historical_QRF is explicitly a different-label-pool baseline audit, not the fair probe comparator',
        tree_structure='Original expanded A trees contain all historical training members; identical probe labels control evidence pool but do not erase original partition provenance'))


if __name__ == '__main__':
    main()

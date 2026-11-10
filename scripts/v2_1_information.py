#!/usr/bin/env python3
"""One fixed forest framework, four matched-band arms; no model search."""
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, DEST, TABLES, load, subset, csv_rows, save_json, sha, atomic_npz
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.probability_v2_1 import psf_admission
from photdis.baseline_v1_2 import DirectClassifier, balanced_weights
from photdis.information_v2_1 import (
    leaf_class_control, fit_calibration, calibrated, class_score, local_neighbors)

ARMS = dict(grz=('g', 'r', 'z'), griz=('g', 'r', 'i', 'z'),
            grzW1=('g', 'r', 'z', 'w1'), grzW1W2=('g', 'r', 'z', 'w1', 'w2'))
MODELS = dict(grz='matched_grz', griz='matched_griz', grzW1='expanded_B', grzW1W2='expanded_C')


def design(t, bands):
    return (np.column_stack([t[b+'0'] for b in bands]),
            np.column_stack([t['sigma_'+b] for b in bands]))


def sample_indices(ids, limit):
    # Deterministic identity-only subset, independent of labels and residuals.
    with np.errstate(over='ignore'):
        rank = np.asarray(ids, dtype='u8')*np.uint64(11400714819323198485)
    return np.argsort(rank, kind='stable')[:limit]


def prepare():
    base = ROOT/'data/baseline_v1_4'
    keys = ['identity', 'ls_id', 'split', 'cell', 'class_label', 'class_eligible', 'Mg',
            'class_train_eligible', 'photometric_eligible', 'identity_conflict', 'nonstellar_risk',
            'redrock_nonstellar_evidence', 'rvs_rr_spectype', 'rvs_quality_reason',
            'rvs_parameter_quality', 'rvs_feh', 'rvs_teff', 'rvs_logg',
            'r', 'optical_error', 'B', 'C', 'i_usable',
            *[b+'0' for b in ('g', 'r', 'i', 'z', 'w1', 'w2')],
            *['sigma_'+b for b in ('g', 'r', 'i', 'z', 'w1', 'w2')]]
    with np.load(base/'references.npz') as z:
        t = {k: z[k] for k in keys}
    native = load(ROOT/'data/v2_0/native_report.npz')
    j = unique_join(t['ls_id'], native['ls_id'])
    types = np.full(len(j), '', dtype='U4')
    types[j >= 0] = native['TYPE'][j[j >= 0]]
    psf, morphology = psf_admission(types)
    t['TYPE'] = types
    matched_ids = None
    inputs = {}
    for name in MODELS.values():
        path = base/'experiments'/f'{name}_identities.npz'
        ids = load(path)['distance_train_identity']
        matched_ids = ids if matched_ids is None else np.intersect1d(matched_ids, ids)
        inputs[str(path.relative_to(ROOT))] = sha(path)
    train = (np.isin(t['identity'], matched_ids) & psf & t['class_train_eligible']
             & t['i_usable'] & t['C'] & (t['split'] == 0))
    # Independent classification admission: NO distance/parallax requirement.
    eval_base = (psf & np.isin(t['split'], [1, 2]) & t['class_eligible']
                 & t['photometric_eligible'] & ~t['identity_conflict'] & ~t['nonstellar_risk'])
    evaluation = eval_base & t['i_usable'] & t['C']
    assert not np.intersect1d(t['identity'][train], t['identity'][evaluation]).size
    raw_control = psf & t['photometric_eligible'] & t['redrock_nonstellar_evidence'] & ~t['identity_conflict']
    save_json(DEST/'information_admission.json', dict(morphology=morphology, training_n=int(train.sum()),
        class_evaluation_before_matched_bands=int(eval_base.sum()), class_evaluation_matched=int(evaluation.sum()),
        class_evaluation_matched_fraction=float(evaluation.sum()/eval_base.sum()),
        inputs=inputs, reference_sha256=sha(base/'references.npz'),
        control_labels=dict(raw_psf_nonstellar=int(raw_control.sum()), confirmed_nonstellar=0,
             status='MWS projection lacks Redrock ZWARN/DELTACHI2; raw SPECTYPE is not a reliable negative control.',
             next_requirement='Acquire assessed Redrock/external QSO/galaxy labels before scientific contamination claim'),
        WD='49 anomalies remain candidates; none used as confirmed WD labels',
        fairness='All four probe arms and controlled leaf-label arms use identical verified-PSF training and evaluation IDs. Frozen partitions were originally fit on different broader distance pools; retain this structural confound.'))
    atomic_npz(DEST/'information_train.npz', **subset(t, train))
    atomic_npz(DEST/'information_eval.npz', **subset(t, evaluation))
    atomic_npz(DEST/'nonstellar_unassessed_psf.npz', **subset(t, raw_control))
    return subset(t, train), subset(t, evaluation)


def groups(t):
    out = {'all': np.ones(len(t['identity']), bool)}
    for c, name in [(0, 'dwarf'), (1, 'evolved'), (2, 'other')]:
        out[name] = t['class_label'] == c
    for lo, hi in [(-np.inf, 18), (18, 20), (20, 22), (22, np.inf)]:
        out[f'r_{lo}_{hi}'] = (t['r'] >= lo) & (t['r'] < hi)
    for lo, hi in [(0, .03), (.03, .1), (.1, np.inf)]:
        out[f'error_{lo}_{hi}'] = (t['optical_error'] >= lo) & (t['optical_error'] < hi)
    for c in np.unique(t['cell']):
        out[f'cell_{c}'] = t['cell'] == c
    return out


def run():
    require_compute()
    TABLES.mkdir(parents=True, exist_ok=True)
    train, test = prepare()
    config = json.loads((ROOT/'configs/baseline_v1_2.json').read_text())
    labels = train['class_label'].astype(int)
    y = test['class_label'].astype(int)
    prior = np.bincount(labels, minlength=3)/len(labels)
    cal = test['split'] == 1
    scores, neighborhood, degradation, calibration = [], [], [], {}
    predictions = dict(identity=test['identity'], TYPE=test['TYPE'], split=test['split'], class_label=y)
    models = {}
    for name, bands in ARMS.items():
        start = time.time()
        tm, te = design(train, bands)
        em, ee = design(test, bands)
        assert np.isfinite(tm).all() and np.isfinite(em).all()
        clf = DirectClassifier(len(bands)-1, 'forest', config).fit(tm, labels, balanced_weights(labels))
        # One fixed algorithm and settings across bands, never select among families.
        joblib.dump(clf, DEST/f'probe_{name}.joblib')
        forest = joblib.load(ROOT/'data/baseline_v1_4/experiments'/f'{MODELS[name]}_distance.joblib')
        qrf, empty = leaf_class_control(forest, tm, labels, em, prior)
        raw = dict(no_color=np.broadcast_to(prior, (len(y), 3)).copy(),
                   frozen_partitions_same_labels=qrf, fixed_forest=clf.predict_proba(em))
        probabilities = {}
        calibration[name] = {}
        for method, prob in raw.items():
            theta = fit_calibration(prob[cal], y[cal])
            calibration[name][method] = theta
            probabilities[method] = calibrated(prob, theta)
            predictions[name+'_'+method] = probabilities[method]
        for split, role in [(1, 'cal_fit'), (2, 'development_test')]:
            for group, mask in groups(test).items():
                use = mask & (test['split'] == split)
                for method in raw:
                    scores.append(dict(bands=name, method=method, split=role, group=group,
                        raw_or_calibrated='calibrated',
                        **class_score(probabilities[method][use], y[use], probabilities['no_color'][use]),
                        empty_frozen_leaf_fraction=float(np.mean(empty[use])) if use.any() else None))
                    scores.append(dict(bands=name, method=method, split=role, group=group,
                        raw_or_calibrated='raw', **class_score(raw[method][use], y[use], raw['no_color'][use])))
        indices = np.flatnonzero(test['split'] == 2)
        indices = indices[sample_indices(test['identity'][indices], 2000)]
        local = local_neighbors(tm, te, labels, train['Mg'], em[indices], ee[indices], y[indices])
        for i, row in enumerate(indices):
            neighborhood.append(dict(bands=name, identity=int(test['identity'][row]),
                class_label=int(y[row]), cell=int(test['cell'][row]), r=test['r'][row], optical_error=test['optical_error'][row],
                **{key: value[i] for key, value in local.items() if value.ndim == 1}))
        models[name] = (clf, forest)
        print('INFORMATION_ARM', name, len(train['identity']), len(y), round(time.time()-start, 1), flush=True)
    # Bright-template measurement degradation with actual weighted target errors.
    # Template identities stay in training; these are conditional simulations,
    # never independent test sources or final-validation replicas.
    parent = load(ROOT/'data/v2_0/parent.npz')
    pp, audit = psf_admission(parent['TYPE'])
    parent = subset(parent, pp)
    rng = np.random.default_rng(20261110)
    high = np.flatnonzero(train['optical_error'] < .02)
    high = high[sample_indices(train['identity'][high], 1000)]
    target = rng.choice(len(parent['identity']), len(high), p=parent['sample_weight']/parent['sample_weight'].sum())
    all_bands = ('g', 'r', 'i', 'z', 'w1', 'w2')
    shared_noise = rng.normal(size=(len(high), len(all_bands)))
    common_route = parent['i_usable'][target] & parent['C'][target]
    common_route &= np.isfinite(design(parent, all_bands)[1][target]).all(axis=1)
    for name, bands in ARMS.items():
        clf, forest = models[name]
        tm, _ = design(train, bands)
        # Target errors are used only where that exact target route is measured.
        route = np.ones(len(target), bool)
        if 'i' in bands:
            route &= parent['i_usable'][target]
        if 'w1' in bands:
            route &= parent['B'][target]
        if 'w2' in bands:
            route &= parent['C'][target]
        _, err = design(parent, bands)
        route &= np.isfinite(err[target]).all(axis=1)
        rows = high[common_route]
        if not len(rows):
            continue
        noise_columns = [all_bands.index(b) for b in bands]
        noisy = tm[rows]+shared_noise[common_route][:, noise_columns]*err[target[common_route]]
        for condition, value in [('high_quality', tm[rows]), ('target_noise', noisy)]:
            pred = calibrated(clf.predict_proba(value), calibration[name]['fixed_forest'])
            base = calibrated(np.broadcast_to(prior, pred.shape), calibration[name]['no_color'])
            degradation.append(dict(bands=name, condition=condition, target_route_retention=float(route.mean()),
                                    common_route_retention=float(common_route.mean()),
                                    template_n=len(high), **class_score(pred, labels[rows], base)))
    # Signed weak flux is retained in the audit; do not replace negative flux
    # with a magnitude or erase it as 'no observation'. Existing weak prototype
    # remains the prediction baseline, not a new tuning target.
    weak = []
    for band in ('W1', 'W2'):
        flux, ivar = parent['FLUX_'+band], parent['FLUX_IVAR_'+band]
        measured = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (parent['NOBS_'+band] > 0)
        state = parent[band.lower()+'_measurement_state']
        for code in range(5):
            use = state == code
            weak.append(dict(band=band, state=code, n=int(use.sum()),
                weighted_fraction=float(np.average(use, weights=parent['sample_weight'])),
                signed_nonpositive=int((use & measured & (flux <= 0)).sum())))
    csv_rows(TABLES/'information_scores.csv', scores)
    csv_rows(TABLES/'color_neighbors.csv', neighborhood)
    csv_rows(TABLES/'template_degradation.csv', degradation)
    csv_rows(TABLES/'weak_flux_inventory.csv', weak)
    save_json(DEST/'probe_calibration.json', calibration)
    atomic_npz(DEST/'information_probabilities.npz', **predictions)
    save_json(DEST/'information_complete.json', dict(job_id=os.getenv('SLURM_JOB_ID'),
        train_n=len(labels), evaluation_n=len(y), parent_morphology=audit,
        covariance='Adjacent independent colors; full shared-band covariance in local neighbors',
        limitations=['Candidate-limited64-neighbor search is not an information upper bound',
                     'Matched i+WISE controls exclude faint/no-WISE objects; report coverage',
                     'Frozen tree partition provenance differs; controlled labels do not undo original tree fitting',
                     'Template noise comparison is conditional on bright training templates, not independent validation',
                     'Weak signed flux inventory alone does not measure weak-flux predictive information'],
        status='Fixed matched-band probe complete; additional quality/signed-flux probability audits tracked separately'))


if __name__ == '__main__':
    run()

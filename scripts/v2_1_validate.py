#!/usr/bin/env python3
"""Actual artifact/identity conservation checks and spatial paired probe audit."""
import json
import os
import numpy as np
from v2_1_near import ROOT, DEST, TABLES, load, csv_rows, save_json, sha
from photdis.stage1 import require_compute


def main():
    require_compute()
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)
    t = load(DEST/'development.npz')
    p = load(DEST/'probability.npz')
    check('probability_identity', np.array_equal(t['identity'], p['identity']))
    check('probability_finite_range', np.isfinite(p['qnear']).all() and ((p['qnear'] >= -1e-7) & (p['qnear'] <= 1+1e-7)).all())
    check('near_class_sum', np.allclose(p['joint_near'][:, :4].sum(axis=1), p['qnear'], atol=2e-6))
    check('near_source_sum', np.allclose(p['joint_near'][:, 4:].sum(axis=1), p['qnear'], atol=2e-6))
    for name in ('development', 'application_psf', 'information_train', 'information_eval',
                 'information_probabilities', 'optical_probe', 'future_validation_reservation',
                 'nonstellar_assessed_psf', 'nonstellar_predictions', 'weak_flux_probability'):
        q = load(DEST/(name+'.npz'))
        check(name+'_PSF', (q['TYPE'] == 'PSF').all())
        check(name+'_unique_identity', len(np.unique(q['identity'])) == len(q['identity']))
    train = load(DEST/'information_train.npz')['identity']
    evaluate = load(DEST/'information_eval.npz')['identity']
    check('matched_probe_train_eval_disjoint', not np.intersect1d(train, evaluate).size)
    opt = load(DEST/'optical_probe.npz')
    check('optical_probe_train_eval_disjoint', not np.intersect1d(opt['train_identity'], opt['identity']).size)
    reserved = load(DEST/'future_validation_reservation.npz')['identity']
    check('reservation_not_development', not np.intersect1d(reserved, t['identity']).size)
    check('reservation_not_probe', not np.intersect1d(reserved, np.r_[train, opt['train_identity'], opt['identity']]).size)
    freeze = json.loads((DEST/'joint_freeze.json').read_text())
    from v2_1_joint import inputs
    check('joint_frozen_inputs_unchanged', inputs()[1] == freeze)
    n = 0
    for path in sorted((DEST/'joint_chunks').glob('*.npz')):
        marker = json.loads(path.with_suffix('.json').read_text())
        check(path.stem+'_sha', sha(path) == marker['sha256'])
        q = load(path)
        check(path.stem+'_PDF_mass', np.allclose(q['pdf'].sum(axis=1), 1, atol=2e-6))
        check(path.stem+'_class_marginal', np.allclose(q['joint'][:, :4].sum(axis=1), q['pdf'], atol=2e-6))
        check(path.stem+'_source_marginal', np.allclose(q['joint'][:, 4:].sum(axis=1), q['pdf'], atol=2e-6))
        check(path.stem+'_identity', np.array_equal(q['identity'], t['identity'][q['rows']]))
        n += len(q['identity'])
    check('all_joint_rows_conserved', n == len(t['identity']))
    # Paired sky-cell bootstrap describes this development footprint only.
    rng = np.random.default_rng(20261110)
    bootrows = []
    for label, data, cells in [
        ('broad_grz', opt, opt['cell']),
        ('matched', load(DEST/'information_probabilities.npz'), load(DEST/'information_eval.npz')['cell'])]:
        roles = data['split'] == 2
        y = data['class_label'][roles]
        cc = cells[roles]
        for arm in ([''] if label == 'broad_grz' else ['grz_', 'griz_', 'grzW1_', 'grzW1W2_']):
            a = data[arm+'fixed_forest'][roles]
            b = data[arm+'frozen_partitions_same_labels'][roles]
            delta = np.log(np.maximum(a[np.arange(len(y)), y], 1e-12))-np.log(np.maximum(b[np.arange(len(y)), y], 1e-12))
            unique = np.unique(cc)
            sums = np.array([delta[cc == c].sum() for c in unique])
            counts = np.array([(cc == c).sum() for c in unique])
            draws = rng.integers(0, len(unique), size=(2000, len(unique)))
            estimates = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
            bootrows.append(dict(scope=label, bands=arm.rstrip('_') or 'grz', n=len(y), cells=len(unique),
                mean_log_gain=float(delta.mean()), positive_cells=int((sums > 0).sum()),
                spatial_bootstrap_low=float(np.quantile(estimates, .025)),
                spatial_bootstrap_high=float(np.quantile(estimates, .975)),
                interpretation='Paired resampling of existing development sky; not new independent sky'))
    csv_rows(TABLES/'probe_spatial_bootstrap.csv', bootrows)
    application = DEST/'application.json'
    app = json.loads(application.read_text())
    check('application_13_DECam_fields', [x['field'] for x in app['catalogs']] == list(range(13)))
    app['scope'] = 'Fixed26000-row stratified sample of13 DECam fields; weighted parent8899334. Northern no-model field excluded. Not all DR11.'
    save_json(application, app)
    save_json(DEST/'validation.json', dict(job_id=os.getenv('SLURM_JOB_ID'), checks=checks,
        actual_checks=len(checks), all_passed=all(checks.values()),
        caveats=['System-membership audit and additional independent references still required before final validation',
                 'Nonstellar labels are conservative pipeline quality, not visual confirmation',
                 'QRF joint mass is reference contribution, not calibrated physical population posterior']))
    print('VALIDATION', len(checks), 'actual checks passed', flush=True)


if __name__ == '__main__':
    main()

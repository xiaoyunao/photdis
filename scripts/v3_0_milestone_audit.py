#!/usr/bin/env python3
"""Recheck actual A–D identities and products before final v3 delivery."""
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT, sha, save_json
from photdis.stage1 import require_compute
from v3_0_validation_inputs import check_freeze

D = ROOT/'data/v3_0'


def main():
    require_compute()
    method = check_freeze()
    inputs, checks = {}, []

    def check(name, condition):
        assert bool(condition), name
        checks.append(name)

    def register(path, expected=None):
        digest = sha(path)
        if expected is not None:
            check('hash:'+str(path.relative_to(ROOT)), digest == expected)
        inputs[str(path.relative_to(ROOT))] = digest
        return path

    def record(path):
        return json.loads(register(path).read_text())

    def arrays(path, names, expected=None):
        with np.load(register(path, expected), allow_pickle=False) as z:
            return {name:z[name] for name in names}

    def identities(name, data):
        ids = data['identity']
        check(name+':int64 unique identities', ids.dtype == np.int64 and len(np.unique(ids)) == len(ids))
        check(name+':native PSF', (data['TYPE'] == 'PSF').all())

    prep = record(D/'preparation.json')
    ref = arrays(D/'references.npz', ['identity','TYPE','role','historical_fit',
        'identity_conflict','primary_channel','specdis_valid','specdis_raw_binary_flag'], prep['output_sha256'])
    identities('reference', ref)
    check('reference admission', len(ref['identity']) == prep['admitted'] == 612114)
    check('reference conflicts excluded', not ref['identity_conflict'].any())
    check('historical fit kept out of calibration/hyper/development',
          not (ref['historical_fit'] & np.isin(ref['role'], [2,3,4])).any())
    check('SpecDis non-1 flags preserved',
          int((ref['specdis_valid'] & (ref['specdis_raw_binary_flag'] != 1)).sum()) == prep['specdis_flag_not1'] == 10298)
    for channel, n in prep['primary_channels'].items():
        check('primary channel '+channel, int((ref['primary_channel'] == int(channel)).sum()) == n)
    cloud_meta = record(D/'sed_cloud.json')
    cloud = arrays(D/'sed_cloud.npz', ['identity','TYPE','rows','labels','mask'], cloud_meta['sha256'])
    identities('SED training', cloud)
    np.testing.assert_array_equal(cloud['identity'], ref['identity'][cloud['rows']])
    check('SED training role', (ref['role'][cloud['rows']] == 1).all())
    check('actual SED training count', len(cloud['identity']) == 32768)
    check('missing class retained', int((cloud['labels'] < 0).sum()) == 5653)
    check('missing FeH retained', int((~cloud['mask'][:,-1]).sum()) == 12270)
    fit = record(D/'sed_fit.json')
    model = joblib.load(register(D/'sed_fit.joblib', fit['sha256']))
    np.testing.assert_array_equal(model.training_identity, cloud['identity'])
    check('nominal fit converged', fit['converged'] and fit['components'] == 128)
    cal = record(D/'calibration.json')
    register(D/'calibration.npz', cal['sha256'])
    for source in ['specdis', 'spdist']:
        t = arrays(D/f'calibration_{source}_inputs.npz', ['identity','TYPE','rows'])
        identities('calibration '+source, t)
        np.testing.assert_array_equal(t['identity'], ref['identity'][t['rows']])
        check('calibration role '+source, len(t['identity']) == 8192 and (ref['role'][t['rows']] == 2).all())
    for draw in range(32):
        base = D/'uncertainty_refit'/f'draw_{draw:02}'
        meta = record(base/'complete.json')
        model = joblib.load(register(base/'model.joblib', meta['sha256']))
        np.testing.assert_array_equal(model.training_identity, cloud['identity'])
        check('full converged shared refit '+str(draw), meta['converged'] and meta['global_draw_id'] == draw)
    control = record(D/'controls/complete.json')
    t = arrays(D/'controls/training.npz', ['identity','TYPE'], control['outputs']['training.npz'])
    identities('QRF training', t)
    np.testing.assert_array_equal(t['identity'], cloud['identity'])
    check('control missing labels retained', control['missing_class_distance_sources'] == 5653)
    for cohort, role in [('hyper',3), ('development',4)]:
        path = D/'cohorts'/(cohort+'.npz')
        t = arrays(path, ['identity','TYPE'])
        identities(cohort, t)
        np.testing.assert_array_equal(np.sort(t['identity']), np.sort(ref['identity'][ref['role'] == role]))
        with np.load(path, allow_pickle=False) as z:
            check(cohort+':no target truth input', not set(z.files) &
                  {'class_label','rvs_feh','p_corr','gaia_parallax','specdis_distance','spdist_distance','Mg'})
    dev = t['identity']
    for view in ['grz_fit', 'griz_fit', 'multiband_fit', 'qrf']:
        base = D/'evaluation_results'/f'development_{view}_0'
        meta = record(base/'complete.json')
        t = arrays(base/'per_star.npz', ['identity','TYPE'], meta['outputs']['per_star.npz'])
        identities('development '+view, t)
        np.testing.assert_array_equal(np.sort(t['identity']), np.sort(dev))
        check('same evaluation count '+view, meta['rows'] == len(dev) == 62880)
        for name, digest in meta['outputs'].items():
            if name.endswith('.csv'):
                register(base/name, digest)
    diagnostics = record(D/'diagnostics/complete.json')
    check('fixed information comparison', diagnostics['n'] == 62880 and diagnostics['known_labels'] == 42950)
    for name, digest in diagnostics['outputs'].items():
        register(D/'diagnostics'/name, digest)
    recovery = record(ROOT/'manifests/v3_0/recovery_final/complete.json')
    check('conditional recovery cases', recovery['catalogs'] == recovery['numerically_passed'] == 224)
    for name, digest in recovery['inputs'].items():
        register(ROOT/name, digest)
    save_json(D/'milestone_delivery_audit.json', dict(passed=True, method_freeze=method,
        checks=checks, check_count=len(checks), inputs=inputs,
        scope='Fresh checks of actual A–D source roles, fitted identities, full refits and matched evaluation products. Scientific calibration and final E–F products require separate checks.',
        source_sha256=sha(ROOT/'scripts/v3_0_milestone_audit.py'), job=os.getenv('SLURM_JOB_ID')))
    print('MILESTONE AUDIT', len(checks), 'checks passed', flush=True)


if __name__ == '__main__':
    main()

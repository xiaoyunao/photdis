#!/usr/bin/env python3
"""Freeze physical choices and scoring before any prospective evidence opens.

Unfinished numerical integrations may continue under the fixed convergence
criteria. They cannot change the physical model or use prospective outcomes to
choose parameters. This is a method freeze, not scientific acceptance.
"""
import json
import os
from v2_1_near import ROOT, sha, save_json
from photdis.stage1 import require_compute

D = ROOT/'data/v3_0'


def main():
    require_compute()
    path = D/'method_freeze.json'
    if path.exists():
        from v3_0_validation_inputs import check_freeze
        print('EXISTING METHOD FREEZE', check_freeze(), flush=True)
        return
    # Only already examined development artifacts are required here.
    recovery = ROOT/'manifests/v3_0/recovery_final/complete.json'
    r = json.loads(recovery.read_text())
    assert r['catalogs'] == r['numerically_passed'] == 224
    target = D/'shared_products/nominal_s2048_native/complete.json'
    t = json.loads(target.read_text())
    assert t['integration_passed'] and t['conditional_marginal_half_delta'] < .005
    paths = [D/'sed_fit.joblib', D/'calibration.npz', D/'uncertainty/shared_draws.npz',
        D/'controls/qrf_grz.joblib', recovery, target,
        ROOT/'docs/v3_0_protocol.md', ROOT/'scripts/v3_0_evaluate.py',
        ROOT/'scripts/v3_0_predict.py', ROOT/'scripts/v3_0_response.py',
        ROOT/'scripts/v3_0_shared_products.py', ROOT/'scripts/v3_0_catalog.py',
        ROOT/'scripts/v3_0_freeze.py', ROOT/'manifests/v3_0/core_validation.json']
    test = json.loads(paths[-1].read_text())
    assert test['passed'] >= 63 and test['failed'] == 0
    paths.extend(sorted((ROOT/'src/photdis/v3').glob('*.py')))
    for draw in range(32):
        model = D/'uncertainty_refit'/f'draw_{draw:02}/model.joblib'
        meta = model.with_name('complete.json')
        info = json.loads(meta.read_text())
        assert info['converged'] and sha(model) == info['sha256']
        paths.extend([model, meta])
    for field in range(14):
        p = D/'catalog'/f'field_{field:02}/prepared.json'
        assert json.loads(p.read_text())['reference_outcomes_read'] is False
        paths.append(p)
    # Photometric identities were registered before reference association.
    paths.extend([ROOT/'data/v2_4/prospective_identity_reservation.npz',
        D/'independent_scale/photometry.npz'])
    paths.extend(D/'new_sky_photometry'/f'field_{i:02}.npz' for i in range(2))
    for name in ('reservation', 'new_sky', 'independent_scale'):
        assert not (D/'validation'/name/'evidence.json').exists(), 'Prospective outcomes already opened'
    record = dict(version='v3.0-method-1', hashes={str(p.relative_to(ROOT)): sha(p) for p in paths},
        primary_method='Frozen empirical native-flux SED, continuous distance, all legal available bands; no target Gaia or spectral evidence in prediction.',
        population=dict(global_logratio_sigma=.7, field_logratio_sigma=.35, sed_tilt_sigma=.3,
            response_draws=list(range(32)), target_case_weight='Design-rescaled at actual sampled size, not expanded parent size',
            prior_roles='Reference-prior and selection-normalized target posterior are separate products; new observed likelihood must not be counted twice.'),
        output=dict(distance_bins_kpc=[0, 2, 5, 10, 20, 50, '>50'], joint='class x distance; preserve binary and SED support, full native tails, modes and qnear',
            counts='int64 realized latent counts; conditional plus shared-parameter plus shared-response covariance; design uncertainty separately'),
        metrics=dict(distance=['median relative bias', 'NMAD', 'abs(delta_mu)>1', '68/95 coverage and width', 'native density log score'],
            gaia=['signed observed predictive log score', 'PIT<0.05', 'PIT>0.95'],
            classes=['log score', 'Brier', 'reliability', 'class-specific recall'],
            mandatory_strata=['2-5kpc', 'red', 'metal-poor', 'other', 'unknown', 'source', 'sky', 'magnitude', 'band availability'],
            reference_scope='Sources are calibrated and dependent evidence, not interchangeable independent ground truth.'),
        numerical=dict(final_joint_half_delta=.005, native_integral=.001, class_distance_cdf_difference=.001,
            max_rhat=1.05, minimum_chain_ess=100, divergences=0,
            bridge_minimum_ess=100, bridge_half_log_evidence=.2, compression_posterior_mass=.001,
            allowed_refinement='More integration points, longer chains and better normalized quadrature proposals; unchanged likelihood, physical priors and scoring. Failures remain explicit.'),
        validation=dict(reservation='3250 identities in 13 already used fields, historical exclusions retained',
            new_sky='2048 photometric-first identities, no-reference objects retained',
            independent_scale='NGC288 HST/LOS dynamical distance 9.03 +0.48/-0.56 kpc; membership-conditioned check, independent of photometric distance method',
            order='Prepare pure photometry, predict, then associate reference outcomes exactly once; no tuning on outcomes'),
        status='Physical method and evaluation freeze; numerical completion and scientific acceptance are separate.', job=os.getenv('SLURM_JOB_ID'))
    save_json(path, record)
    print('METHOD FROZEN', sha(path), flush=True)


if __name__ == '__main__':
    main()

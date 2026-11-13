"""Independent density identities and mechanism controls on analytic mixtures."""
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from photdis.v3.inference import reference_log_density, target_posterior
from photdis.v3.likelihood import normalize_joint
from photdis.v3_1.diagnostics import condition_feh, control_log_density, infer_control, mode_attribution
from test_v3_inference import model, observation


def test_reference_control_preserves_frozen_density_and_target_prior():
    m, o = model(), observation(); grid = np.linspace(0, 30, 601)
    lp = control_log_density(m, o, grid)
    np.testing.assert_allclose(logsumexp(lp, axis=0),
                               logsumexp(reference_log_density(m, o, grid), axis=0), atol=1e-12)
    theta = np.arange(1, 16, dtype=float).reshape(3, 5); theta /= theta.sum()
    p, _ = target_posterior(m, o, grid, theta)
    new = normalize_joint(grid, control_log_density(m, o, grid, theta=theta))
    np.testing.assert_allclose(new.joint_density, p.joint_density, atol=1e-14)


def test_feh_conditioning_matches_independent_joint_gaussian():
    m = model(); m.covariances[0, 0, 6] = m.covariances[0, 6, 0] = .7
    original_mean = m.means.copy(); original_cov = m.covariances.copy()
    value, error = -.1, .2
    new, logweight = condition_feh(m, value, error)
    observed = np.r_[m.means[0, :6]+np.arange(6), value]
    cov = m.covariances[0].copy(); cov[6, 6] += error**2
    joint = multivariate_normal.logpdf(observed, mean=m.means[0], cov=cov)
    conditional = multivariate_normal.logpdf(observed[:6], mean=new.means[0, :6], cov=new.covariances[0, :6, :6])
    np.testing.assert_allclose(joint, conditional+logweight[0], atol=1e-12)
    assert new.means[0, 0] > m.means[0, 0]
    np.testing.assert_array_equal(m.means, original_mean)
    np.testing.assert_array_equal(m.covariances, original_cov)


def test_class_oracle_is_normalized_and_not_unknown_as_physical_class():
    r = infer_control(model(), observation(), reference_class=1, initial_step=.2)
    np.testing.assert_allclose(r['class_distance'].sum(axis=1), [0, 1, 0], atol=1e-12)
    assert r['diagnostic_only'] and r['posterior'].convergence['converged']
    with pytest.raises(ValueError):
        control_log_density(model(), observation(), np.arange(5), reference_class=-1)


def test_modes_conserve_joint_class_binary_and_identify_binary_separation():
    m = model(); m.weights = np.array([1., 0.]); m.classes[0] = [1., 0., 0.]
    result = infer_control(m, observation(), initial_step=.1)
    modes = result['modes']
    np.testing.assert_allclose(sum(x['mass'] for x in modes), 1., atol=1e-12)
    np.testing.assert_allclose(sum(x['mass']*x['binary_probability'] for x in modes), result['binary_probability'], atol=1e-12)
    significant = [x for x in modes if x['mass'] > .05]
    assert len(significant) == 2
    assert abs(significant[1]['mu']-significant[0]['mu']-2.5*np.log10(2)) < .05
    assert significant[0]['binary_probability'] < .01 and significant[1]['binary_probability'] > .99
    assert all(x['class_probability'][0] > .999 for x in significant)


def test_invalid_controls_fail_and_signed_flux_survives():
    m, o = model(), observation(); old = o['flux'].copy()
    result = infer_control(m, o, error_scale=np.ones(6)*2, initial_step=.2)
    np.testing.assert_array_equal(o['flux'], old)
    assert np.isfinite(result['full_logpdf']).all()
    with pytest.raises(ValueError): condition_feh(m, np.nan, .1)
    with pytest.raises(ValueError): control_log_density(m, o, np.arange(5), reference_feh=-1.)
    with pytest.raises(ValueError): control_log_density(m, o, np.arange(5), error_scale=np.zeros(6))
    with pytest.raises(ValueError): mode_attribution(result['posterior'], 1)

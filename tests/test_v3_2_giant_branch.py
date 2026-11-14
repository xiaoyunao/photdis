import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_2.giant_branch import adjust_giant_branch
from photdis.v3_2.giant_branch import reference_class_curves
from photdis.v3.inference import kernel_loglike
from photdis.v3_1.shared_prior import marginalize_native
from test_v3_inference import model, observation
from v3_2_branch_forecast import forecast_level
from v3_2_branch_development import summarize


def test_adjustment_changes_only_giant_odds_and_preserves_conditional_distance():
    grid = np.linspace(-20, 50, 701)
    shape = norm.logpdf(grid[None], [[8], [17], [10]], [[1], [.4], [2]])
    log_width = np.log(trapezoid_weights(grid))
    shape -= logsumexp(shape + log_width, axis=1, keepdims=True)
    target = np.array([.8, .1, .1])
    reference = np.array([.2, .5, .3])
    adjusted = adjust_giant_branch(grid, shape + np.log(target)[:, None], shape, reference)
    log_mass = logsumexp(adjusted + log_width, axis=1)
    mass = np.exp(log_mass)
    np.testing.assert_allclose(mass.sum(), 1)
    np.testing.assert_allclose(mass[1]/(1-mass[1]), 1/3)
    np.testing.assert_allclose(mass[0]/mass[2], 8)
    np.testing.assert_allclose(adjusted - log_mass[:, None], shape, atol=1e-12)


def test_log_odds_midpoint_retains_tiny_positive_class_support():
    grid = np.arange(4, dtype=float)
    target = np.broadcast_to(np.array([0., -800., -2.])[:, None], (3, 4))
    adjusted = adjust_giant_branch(grid, target, np.zeros_like(target), [1., 1., 1.])
    assert np.isfinite(adjusted).all()
    assert -410 < adjusted[1, 0] < -390


def test_external_forecast_reproduces_adjustment_from_native_joint_density():
    sed, obs = model(), observation()
    grid = np.linspace(-20, 50, 3501)
    cal = np.zeros(6)
    theta = np.full((4, 3, 5), 1/15)
    beta = np.zeros((4, 2))
    weights = np.full(4, .25)
    lk, _ = kernel_loglike(sed, obs, grid, calibration_mag=cal)
    base = marginalize_native(sed, grid, lk, theta, beta, weights)
    density = base['posterior'].joint_density.reshape(2, len(sed.weights), 3, len(grid))
    native = logsumexp(lk + np.log([.9, .1])[:, None, None], axis=0)
    with np.errstate(divide='ignore'):
        expected_curve = adjust_giant_branch(
            grid, np.log(density.sum(axis=(0, 1))),
            reference_class_curves(sed, grid, native), sed.weights @ sed.classes)
    _, joint, summary = summarize(grid, expected_curve, dict(
        class_label=-1, parallax_eligible=False, distance_available=False))
    _, quantiles, actual, passed = forecast_level(sed, obs, cal, grid, theta, beta, weights)
    assert passed
    np.testing.assert_allclose(actual, joint, atol=1e-12, rtol=0)
    np.testing.assert_allclose(quantiles, summary['mu'], atol=1e-10, rtol=0)

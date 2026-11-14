import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_2.giant_branch import adjust_giant_branch


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

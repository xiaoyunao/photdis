import numpy as np
import pytest
from scipy.stats import norm
from photdis.v3_1.luminosity import luminosity_transform


def test_achromatic_likelihood_gaussian_shift_and_variance():
    grid = np.linspace(-10, 10, 4001)
    original = norm.pdf(grid, 0.4, .7)
    actual = luminosity_transform(original, grid, .31, .42)
    expected = norm.pdf(grid, .71, np.hypot(.7, .42))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-13)
    np.testing.assert_array_equal(luminosity_transform(original, grid, 0., 0.), original)


def test_luminosity_operation_precedes_radial_prior_and_is_linear():
    grid = np.linspace(-10, 10, 4001)
    likelihood = np.array([norm.pdf(grid, -1, .5), norm.pdf(grid, 2, 1.)])
    weights = np.array([.2, .8]); prior = norm.pdf(grid, 0., 1.)
    transformed = luminosity_transform(likelihood, grid, .2, .3)
    np.testing.assert_allclose(weights@transformed, luminosity_transform(weights@likelihood, grid, .2, .3), atol=1e-14)
    correct = transformed*prior
    incorrect = luminosity_transform(likelihood*prior, grid, .2, .3)
    assert np.max(abs(correct-incorrect)) > .01
    # Exactly the same operation applies to entry probability curves.
    entry = luminosity_transform(np.ones_like(likelihood), grid, .2, .3)
    np.testing.assert_allclose(entry[:, (grid > -5) & (grid < 5)], 1., atol=1e-14)


def test_luminosity_rejects_unregistered_support():
    grid = np.linspace(-3, 3, 31)
    with pytest.raises(ValueError): luminosity_transform(np.ones(31), grid, 1.01, .1)
    with pytest.raises(ValueError): luminosity_transform(np.ones(31), grid**3, 0., .1)

import numpy as np
from scipy.special import ndtr

from photdis.parallax_forward import (interval_selected_pit,
                                      selected_parallax_null_draws)


def test_pit_conditions_on_both_group_boundaries():
    lower = np.array([0., .2, .7])
    upper = np.array([.2, .7, .95])
    y = np.array([.1, .45, .825])
    pit, probability = interval_selected_pit(y, lower, upper)
    np.testing.assert_allclose(pit, .5)
    np.testing.assert_allclose(probability, [.2, .5, .25])


def test_null_draws_respect_observed_selection():
    # Mg=5 and g0=15 imply a 1 mas component, each with 1 mas noise.
    samples = selected_parallax_null_draws(
        np.array([1.]), np.array([15.]), np.array([5.]),
        np.array([[1.]]), seed=7, n_sim=3000, selection_snr_max=1)
    assert samples.shape == (3000, 1)
    assert np.all(samples < 1)
    expected_mean = 1 - np.exp(-.5 * 0**2) / np.sqrt(2*np.pi) / ndtr(0)
    assert abs(float(samples.mean()) - expected_mean) < .06

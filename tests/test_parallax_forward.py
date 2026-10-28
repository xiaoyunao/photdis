import numpy as np
import pytest

from photdis.parallax_forward import (parallax_forward_check,
                                     weighted_selected_parallax_check)


def test_negative_observed_parallax_is_retained_in_forward_check():
    result = parallax_forward_check(
        np.array([-1., 1.]), np.array([.5, .5]), np.array([15., 15.]),
        np.tile([5., 5., 5., 5., 5.], (2, 1)))
    assert np.allclose(result["model_parallax_mean_mas"], 1.)
    assert result["approximate_pit"][0] < result["approximate_pit"][1]
    assert np.isfinite(result["standardized_residual"]).all()


def test_forward_check_rejects_unordered_quantiles():
    with pytest.raises(ValueError):
        parallax_forward_check([1.], [.1], [15.], [[5., 6., 4., 7., 8.]])


def test_selected_pit_conditions_on_observed_parallax_cut():
    # Mg=5 at g0=15 means a 1 mas model parallax. At sigma=.1, the cut
    # Y<10 sigma is at the center of its Gaussian observed distribution.
    result = weighted_selected_parallax_check(
        [.95, -.1], [.1, .1], [15., 15.], [5.], [[1.], [1.]])
    assert np.allclose(result["model_parallax_mean_mas"], 1)
    assert np.allclose(result["selection_probability"], .5)
    assert np.allclose(result["selected_pit"], 2 * result["unconditional_pit"])
    assert result["selected_pit"][1] < result["selected_pit"][0]


def test_selected_pit_rejects_rows_not_in_selected_sample():
    with pytest.raises(ValueError, match="upper selection"):
        weighted_selected_parallax_check([1.], [.1], [15.], [5.], [[1.]])

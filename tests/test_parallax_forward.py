import numpy as np
import pytest

from photdis.parallax_forward import parallax_forward_check


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

import numpy as np

from photdis.giant_kernel import (color_covariance, conditional_weights,
                                  mixture_cdf, mixture_logpdf, predict_one)


def test_color_covariance_includes_shared_r_band():
    cov = color_covariance(.03, .04, .05)
    np.testing.assert_allclose(cov, [[.03**2 + .04**2, -.04**2],
                                     [-.04**2, .04**2 + .05**2]])


def test_conditional_mixture_retains_two_luminosity_solutions():
    colors = np.array([[.8, .4], [.8, .4]])
    covariance = np.broadcast_to(np.eye(2) * .001, (2, 2, 2))
    prediction = predict_one(colors, covariance, [-2., 2.], [.1, .1],
                             [.8, .4], np.eye(2) * .001, .1, .1)
    np.testing.assert_allclose(prediction["effective_n"], 2)
    assert -1 < prediction["quantiles"][2] < 1
    weight, _ = conditional_weights(colors, covariance, [.8, .4],
                                    np.eye(2) * .001, .1)
    sigma = np.full(2, np.hypot(.1, .1))
    np.testing.assert_allclose(
        mixture_cdf(prediction["quantiles"][0], np.array([-2., 2.]), sigma,
                    weight), .025, atol=1e-5)


def test_log_density_ignores_underflowed_zero_kernel_weights():
    value = mixture_logpdf(0., np.array([0., 20.]), np.array([1., 1.]),
                           np.array([1., 0.]))
    np.testing.assert_allclose(value, -.5 * np.log(2 * np.pi))

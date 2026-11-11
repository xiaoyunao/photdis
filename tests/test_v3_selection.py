from types import SimpleNamespace
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from scipy.stats import norm
from photdis.v3.selection import component_entry, integrate_entry
from photdis.v3.inference import DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA


def test_entry_matches_native_flux_gaussian_threshold_and_continuous_integral():
    mean = np.array([[100., 120., 130., 150., 160., 170., -.5]])
    model = SimpleNamespace(means=mean, covariances=np.array([np.eye(7)*1e-12]),
        weights=np.ones(1), classes=np.ones((1, 3))/3, binary_reference_fraction=.1,
        binary_brightness_factor=2.)
    error = np.array([.1, .2, .15]); mu = np.linspace(-15, 40, 1101)
    response = component_entry(model, error, mu, draws=32)
    expected = ndtr(mean[0, [0, 1, 3]][None, :]*10**(-.4*(mu[:, None]-10))/error-5).prod(axis=1)
    np.testing.assert_allclose(response[0, 0], expected, atol=1e-8)
    result = integrate_entry(model, response, mu)
    for d, (center, sigma) in enumerate(zip(DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA)):
        def integrand(m):
            p = 0.
            for brightness, weight in [(1, .9), (2, .1)]:
                p += weight*ndtr(mean[0, [0, 1, 3]]*brightness*10**(-.4*(m-10))/error-5).prod()
            return p*norm.pdf(m, center, sigma)
        truth = quad(integrand, -15, 40, epsabs=1e-10)[0]
        np.testing.assert_allclose(result[:, d], truth, atol=1e-7)

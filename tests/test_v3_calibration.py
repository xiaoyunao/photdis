import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from photdis.v3.calibration import signed_parallax_loglike, reference_nodes


def test_signed_parallax_matches_independent_adaptive_integral():
    mu = np.array([12., 13., 14.]); sigma = np.array([.3, .2, .4])
    p = np.array([-.2, .3, .15]); error = np.array([.3, .1, .001])
    out = signed_parallax_loglike(mu, sigma, p, error, offset=.1, extra=.05, order=96)
    for j in range(3):
        s = np.hypot(sigma[j], .05); center = mu[j]+.1
        if error[j] < .01:
            value = quad(lambda x: norm.pdf(x, p[j], error[j])*norm.pdf(10-5*np.log10(x), center, s)*5/(np.log(10)*x),
                         p[j]-10*error[j], p[j]+10*error[j], epsabs=1e-12)[0]
        else:
            value = quad(lambda x: norm.pdf(x, center, s)*norm.pdf(p[j], 10**((10-x)/5), error[j]),
                         center-10*s, center+10*s, epsabs=1e-12)[0]
        assert out[j] == pytest.approx(np.log(value), abs=1e-6)


def test_gaia_negative_parallax_nodes_and_vac_broad_tail():
    mu, w = reference_nodes(np.array([1, 3]), np.array([2., np.nan]), np.array([.2, np.nan]),
                           np.array([1., -.3]), np.array([.1, .2]), nodes=256, outlier=.1)
    assert np.isfinite(mu).all() and np.all(np.diff(mu, axis=1) > 0)
    assert mu[0, -1]-mu[0, 0] > 5
    np.testing.assert_allclose(w.sum(axis=1), 1)
    assert np.median(mu[1]) > 12


def test_precise_gaia_nodes_resolve_sub_grid_likelihood():
    mu, _ = reference_nodes(np.array([3]), np.array([np.nan]), np.array([np.nan]),
                           np.array([1.123]), np.array([.0001]), nodes=128)
    expected = 10-5*np.log10(1.123)
    assert np.median(mu[0]) == pytest.approx(expected, abs=1e-6)
    assert np.std(mu[0]) == pytest.approx(5/np.log(10)*.0001/1.123, rel=.02)


def test_tiny_broad_reference_mass_is_not_lost_or_boundary_truncated():
    mu, w = reference_nodes(np.array([1]), np.array([1000.]), np.array([.1]),
                           np.array([-.01]), np.array([.1]), nodes=24, outlier=.001)
    assert mu.max() > 30
    center = 25.
    assert (w*mu).sum() == pytest.approx(center, abs=1e-10)
    expected_variance = .999*.1**2+.001*(2.**2+.1**2)
    assert (w*(mu-center)**2).sum() == pytest.approx(expected_variance, abs=1e-10)

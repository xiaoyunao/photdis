import numpy as np
import pytest
from scipy.stats import norm, multivariate_normal
from photdis.v3.likelihood import (flux_loglikelihood, adaptive_posterior,
    reference_flux_covariance, intrinsic_covariance, normalize_joint)


def test_native_flux_matches_independent_multivariate_forward_model():
    f = np.array([5., -1., 8.]); e = np.array([.2, .5, .3])
    sed = np.array([[100., 80., 140.], [20., 30., 40.]])
    a = np.array([[.2, .1, .1], [.1, .3, -.05], [.1, -.05, .4]])
    cov = np.stack([a, 2*a]); tr = np.array([.9, .95, .99])
    mu = np.array([13.1, 17.7])
    actual = flux_loglikelihood(f, e, [1, 1, 1], tr, sed, cov, mu)
    for k in range(2):
        for j, m in enumerate(mu):
            transform = np.diag(tr*10**(-.4*(m-10)))
            mean = transform@sed[k]
            variance = transform@cov[k]@transform.T+np.diag(e**2)
            assert actual[k, j] == pytest.approx(multivariate_normal.logpdf(f, mean, variance), abs=1e-11)
    omitted = flux_loglikelihood([5, np.nan, 8], [.2, np.nan, .3], [1, 0, 1], tr, sed, cov, mu)
    reduced = flux_loglikelihood(f[[0, 2]], e[[0, 2]], [1, 1], tr[[0, 2]], sed[:, [0, 2]], cov[:, [0, 2]][:, :, [0, 2]], mu)
    np.testing.assert_allclose(omitted, reduced)
    with pytest.raises(ValueError, match='system mismatch'):
        flux_loglikelihood(f, e, [1]*3, tr, sed, cov, mu, system='BASS_MzLS_DR9')


def test_censored_flux_uses_conditional_probability_and_signed_detection():
    sed = np.array([[2., 3.]])
    cov = np.array([[[1., .4], [.4, 2.]]])
    observed = np.array([-1., 2.]); e = np.array([.5, .8])
    result = flux_loglikelihood(observed, e, [1, 1], [1, 1], sed, cov, [10.], upper_limit=[0, 1])[0, 0]
    v = cov[0]+np.diag(e**2)
    expected = norm.logpdf(-1, 2, np.sqrt(v[0, 0]))
    conditional_mean = 3+v[1, 0]/v[0, 0]*(-1-2)
    conditional_var = v[1, 1]-v[1, 0]**2/v[0, 0]
    expected += norm.logcdf(2, conditional_mean, np.sqrt(conditional_var))
    assert result == pytest.approx(expected)


def test_continuous_off_node_multimodal_distance_with_far_tail():
    def likelihood(mu):
        return np.array([np.log(.6)+norm.logpdf(mu, 11.27, .13),
                         np.log(.4)+norm.logpdf(mu, 19.21, .22)])
    p = adaptive_posterior(likelihood, step=.1)
    assert p.convergence['converged']
    np.testing.assert_allclose(p.component_mass(), [.6, .4], atol=1e-7)
    assert p.cdf(15).sum() == pytest.approx(.6, abs=1e-7)
    assert p.distance_bins()[:, -1].sum() > .39
    modes, mass = p.modes()
    np.testing.assert_allclose(modes, [11.27, 19.21], atol=.015)
    np.testing.assert_allclose(mass, [.6, .4], atol=1e-7)
    assert p.distance_bins().sum() == pytest.approx(1)


def test_quadrature_extends_boundary_and_normalizes_nonuniform_grid():
    p = adaptive_posterior(lambda x: norm.logpdf(x, 26, .3)[None], step=.15)
    assert p.convergence['extensions'] > 0 and p.convergence['converged']
    assert p.quantiles()[2] == pytest.approx(26, abs=.01)
    q = normalize_joint([0, 1, 3], np.log([[1, 1, 1]]))
    assert q.cdf(.5).sum() == pytest.approx(1/6)
    assert q.cdf(2).sum() == pytest.approx(2/3)


def test_reference_distance_error_has_shared_band_covariance():
    f = np.array([10., 20., 30.]); e = np.array([.1, .2, .3]); sig = .2
    cov = reference_flux_covariance(f, e, sig)
    common = .4*np.log(10)*f*sig
    np.testing.assert_allclose(cov, np.diag(e**2)+np.outer(common, common))
    # A synthetic empirical cloud with known additive measurement covariance.
    rng = np.random.default_rng(1)
    latent = np.array([[1, .3], [.3, 2.]])
    noise = np.array([[.4, .2], [.2, .5]])
    samples = rng.multivariate_normal([0, 0], latent+noise, size=100000)
    _, fitted, _ = intrinsic_covariance(samples, np.broadcast_to(noise, (len(samples), 2, 2)))
    np.testing.assert_allclose(fitted, latent, atol=.03)


def test_small_dust_uncertainty_keeps_cross_band_covariance():
    f = np.array([1., 2., 3.]); error = np.array([.1]*3)
    sed = np.array([[10., 20., 30.]]); cov = np.array([np.eye(3)])
    coefficient = np.array([3., 2., 1.]); sigma_ebv = .003; mu = 12.5
    actual = flux_loglikelihood(f, error, [1]*3, [1]*3, sed, cov, [mu],
                              dust_sigma_ebv=sigma_ebv, extinction_coefficients=coefficient)[0, 0]
    a = 10**(-.4*(mu-10)); mean = a*sed[0]
    direction = .4*np.log(10)*sigma_ebv*coefficient*mean
    variance = a*a*cov[0]+np.diag(error**2)+np.outer(direction, direction)
    expected = multivariate_normal.logpdf(f, mean, variance)
    assert actual == pytest.approx(expected, abs=1e-11)

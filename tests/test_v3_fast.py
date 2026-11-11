import numpy as np
from scipy.stats import multivariate_normal
from photdis.v3.fast import spectral_flux_loglikelihood
from photdis.v3.likelihood import flux_loglikelihood


def test_spectral_likelihood_against_native_multivariate_gaussian():
    rng = np.random.default_rng(918)
    means = np.exp(rng.normal(3, 1, (9, 6)))
    a = rng.normal(size=(9, 6, 6))*.2
    cov = (a@a.transpose(0, 2, 1))*means[:, :, None]*means[:, None, :]
    flux = np.array([1., .4, np.nan, 1.2, -.8, -.1])
    error = np.array([.01, .2, np.nan, .04, 1., .8])
    mask = np.array([1, 1, 0, 1, 1, 1], bool)
    trans = np.array([.89, .93, np.nan, .96, .99, .995])
    coefficient = np.array([3., 2., 1.5, 1., .2, .1])
    calibration = np.array([.01, -.01, 0, .005, 0, .01])
    mu = np.linspace(3, 27, 111)
    kwargs = dict(dust_sigma_ebv=.004, extinction_coefficients=coefficient,
                  calibration_mag=calibration)
    got = spectral_flux_loglikelihood(flux, error, mask, trans, means, cov, mu, **kwargs)
    direct = flux_loglikelihood(flux, error, mask, trans, means, cov, mu, **kwargs)
    np.testing.assert_allclose(got, direct, rtol=1e-9, atol=1e-8)
    # Independently construct the observed native-band Gaussian, including dust.
    for k, j in [(0, 10), (4, 55), (8, 90)]:
        factor = 10**(-.4*(mu[j]-10))*trans[mask]*10**(-.4*calibration[mask])
        m = means[k, mask]*factor
        v = cov[k][np.ix_(mask, mask)]*factor[:, None]*factor[None, :]
        d = .4*np.log(10)*.004*coefficient[mask]*m
        v += np.diag(error[mask]**2)+np.outer(d, d)
        np.testing.assert_allclose(got[k, j], multivariate_normal.logpdf(flux[mask], m, v), rtol=1e-9)


def test_upper_limit_falls_back_to_integrated_event():
    mean = np.array([[2., 3.]])
    cov = np.array([[[.3, .1], [.1, .4]]])
    args = ([1., 2.], [.1, .2], [True, True], [1., 1.], mean, cov, [10., 11.])
    kw = dict(upper_limit=[False, True])
    np.testing.assert_allclose(spectral_flux_loglikelihood(*args, **kw), flux_loglikelihood(*args, **kw))

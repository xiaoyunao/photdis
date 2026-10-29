import numpy as np

from photdis.xd_distance import XDMixture, measurement_covariance


def test_training_covariance_preserves_shared_g_r_and_parallax():
    sigma = np.array([[.02, .03, .04, .05, .06]])
    cov = measurement_covariance(sigma, np.array([.01]))[0]
    assert cov.shape == (5, 5)
    assert np.isclose(cov[0, 1], .02**2)
    assert np.isclose(cov[1, 2], -.03**2)
    assert np.isclose(cov[2, 3], -.04**2)
    assert np.isclose(cov[3, 4], -.05**2)
    assert np.isclose(cov[0, 0], .02**2 + (5/np.log(10)*.01)**2)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_single_gaussian_distance_integrates_correlated_errors():
    model = XDMixture(1)
    model.weights_ = np.array([1.])
    model.means_ = np.array([[8., 1., .5]])
    model.covariances_ = np.array([[[.25, .08, 0],
                                    [.08, .04, 0],
                                    [0, 0, .01]]])
    observed = np.array([[18., 1., .5]])
    error = measurement_covariance(np.array([[.02, .03, .04]]))
    quantiles = model.posterior_mu(observed, error)
    assert quantiles.shape == (1, 5)
    assert np.all(np.diff(quantiles[0]) > 0)
    assert abs(quantiles[0, 2] - 10) < 1e-6
    assert 0.25 < quantiles[0, 3] - 10 < 0.4


def test_xd_fit_scores_heldout_conditional_density():
    rng = np.random.default_rng(9)
    x = np.column_stack((rng.normal(8, .3, 120),
                         rng.normal(1.2, .1, 120),
                         rng.normal(.6, .05, 120)))
    cov = measurement_covariance(np.full((120, 3), .02),
                                  np.full(120, .02))
    model = XDMixture(2, max_iter=8, batch_size=32).fit(x, cov)
    assert np.isclose(model.weights_.sum(), 1)
    assert np.linalg.eigvalsh(model.covariances_).min() > 0
    assert np.isfinite(model.conditional_mg_logpdf(x[:8], cov[:8])).all()

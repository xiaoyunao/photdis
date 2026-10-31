import numpy as np
from scipy.stats import norm

from photdis.giant_information import (
    colors_and_covariance, band_realizations, linked_groups, wilson, purity,
    class_metrics, fit_classifier, predict_classifier, fit_abundance,
    predict_abundance, abundance_metrics, abundance_error, field_bootstrap)


def test_shared_colors_and_unused_missing_bands():
    m = np.array([[18., 17., 16.5, np.nan, np.nan]])
    e = np.array([[.02, .03, .04, np.nan, np.nan]])
    c, v = colors_and_covariance(m, e, 2)
    np.testing.assert_allclose(c, [[1., .5]])
    np.testing.assert_allclose(v[0], [[.0013, -.0009], [-.0009, .0025]])
    m[0, 3:] = [16., 16.1]; e[0, 3:] = [.05, .06]
    draws = band_realizations(m, e, draws=100000, seed=10)
    colors, _ = colors_and_covariance(draws, np.zeros_like(draws), 4)
    nominal, theoretical = colors_and_covariance(m, e, 4)
    np.testing.assert_allclose(colors.mean(axis=1), nominal, atol=1e-12)
    v = theoretical[0]
    # Antithetic pairs provide 50,000 independent variance realizations.
    standard_error = np.sqrt((np.outer(v.diagonal(), v.diagonal())+v*v)/49999)
    assert np.all(np.abs(np.cov(colors[0].T)-v)<5*standard_error)


def test_transitive_identity_links_and_missing_ids():
    groups = linked_groups(np.array([1, 2, 3, 4, 5]), np.array([100, 100, 0, 0, 0]),
                           np.array([b'a', b'b', b'b', b'', b'']))
    assert groups[0] == groups[1] == groups[2]
    assert len(set(groups)) == 3
    assert groups[3] != groups[4]


def test_classifier_retains_class_balance_and_same_draws():
    rng = np.random.default_rng(17)
    labels = np.r_[np.zeros(100, int), np.ones(100, int)]
    m = rng.normal(17, .05, (200, 5))
    m[:, 0] += labels*.7
    e = np.full_like(m, .02)
    model = fit_classifier(m, e, labels, 2, seed=19)
    score, noise, outside = predict_classifier(model, m, e, seed=20)
    metrics = class_metrics(labels, score)
    assert metrics['giants']['rate'] > .95 and metrics['dwarfs']['rate'] < .05
    assert not outside.any() and np.all(noise>=0)
    m[:, 3:] = np.nan; e[:, 3:] = np.nan
    repeated, _, _ = predict_classifier(model, m, e, seed=20)
    np.testing.assert_array_equal(score, repeated)


def test_abundance_forward_error_and_hidden_test_label():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(600, 2))
    v = np.broadcast_to(np.array([[.002, -.001], [-.001, .003]]), (600, 2, 2)).copy()
    y = -1.4 + .6*x[:, 0] - .2*x[:, 1] + rng.normal(0, .22, 600)
    sy = np.full(600, .1)
    model = fit_abundance(x[:400], v[:400], y[:400], sy[:400])
    mean, sigma = predict_abundance(model, x[400:], v[400:])
    np.testing.assert_allclose(model['beta'][1:]/model['scale'], [.6, -.2], atol=.06)
    report, logp = abundance_metrics(y[400:], sy[400:], mean, sigma)
    np.testing.assert_allclose(logp, norm.logpdf(y[400:], mean, np.hypot(sigma, sy[400:])))
    assert .85 < report['coverage95']['rate'] <= 1
    baseline = fit_abundance(np.empty((400, 0)), np.empty((400, 0, 0)), y[:400], sy[:400])
    bm, bs = predict_abundance(baseline, np.empty((200, 0)), np.empty((200, 0, 0)))
    _, bl = abundance_metrics(y[400:], sy[400:], bm, bs)
    assert np.mean(logp-bl) > .5
    # Prediction accepts no target FeH; changing the score's label changes
    # its log density, never the frozen conditional prediction.
    changed, _ = predict_abundance(model, x[400:], v[400:])
    np.testing.assert_array_equal(mean, changed)


def test_reference_error_derivative_and_rate_uncertainty():
    from photdis.audit import rvs_metallicity
    t = np.array([4300., 4800.]); te = np.array([40., 70.])
    sigma, random = abundance_error(np.array([.1, .2]), t, te)
    plus = rvs_metallicity(np.array([-1., -1.]), t+.01, np.array([2., 2.]))[0]
    minus = rvs_metallicity(np.array([-1., -1.]), t-.01, np.array([2., 2.]))[0]
    derivative = (plus-minus)/.02
    np.testing.assert_allclose(random, np.sqrt(np.array([.1, .2])**2+(derivative*te)**2), rtol=1e-8)
    np.testing.assert_allclose(sigma**2-random**2, .01)
    assert wilson(0, 10)['interval95'][1] > .2
    assert wilson(0, 0)['rate'] is None
    np.testing.assert_allclose(purity(.8, .05, [.01]), [.008/(.008+.0495)])
    assert purity(0., 0., [.1]) == [None]
    assert field_bootstrap([], seed=1)['mean'] is None

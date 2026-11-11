import itertools
import numpy as np
from scipy.stats import norm
from photdis.v3.likelihood import normalize_joint
from photdis.v3.compression import compress
from photdis.v3.metrics import observed_parallax, reference_mu_score, normal_reference_score, design_count_covariance


def test_reference_score_matches_independent_normal_convolution():
    grid = np.linspace(8, 18, 4001)
    lp = norm.logpdf(grid, 13., .4)
    p = normalize_joint(grid, np.stack([lp+np.log(.9), lp+np.log(.1)]))
    c = compress(p, 1, tolerance=1e-5)
    out = reference_mu_score(c, 13.2, .3)
    np.testing.assert_allclose(out['log_score'], norm.logpdf(13.2, 13, .5), atol=2e-4)
    np.testing.assert_allclose(out['pit'], norm.cdf(.2/.5), atol=2e-5)
    for truth in [11., 13.2, 16.]:
        np.testing.assert_allclose(normal_reference_score(grid, lp, truth, .3),
                                   norm.logpdf(truth, 13, .5), atol=2e-5)
    for parallax in [-.1, .2, .5]:
        a = observed_parallax(grid, lp, parallax, .01, order=8)
        b = observed_parallax(grid, lp, parallax, .01, order=32)
        np.testing.assert_allclose([a['pit'], a['log_score']], [b['pit'], b['log_score']], atol=1e-7)


def test_design_covariance_agrees_with_enumerating_all_samples():
    parent = np.array([[1., 0.], [.7, .3], [.4, .6], [0., 1.]])
    estimates = []; variances = []
    for indices in itertools.combinations(range(4), 2):
        total, covariance = design_count_covariance(parent[list(indices)], [0, 0], {0: 4})
        estimates.append(total); variances.append(covariance)
    estimates = np.asarray(estimates); delta = estimates-parent.sum(axis=0)
    np.testing.assert_allclose(np.mean(variances, axis=0), delta.T@delta/len(delta))


def test_native_histogram_scores_preserve_discontinuities_and_signed_parallax():
    from scipy.integrate import quad
    from photdis.v3.metrics import histogram_parallax, histogram_reference_score
    edges = np.array([8., 11., 11.1, 16., 20.]); mass = np.array([.1, .6, 0., .3])
    for p in [-.2, .3, .8]:
        direct = sum(m/(b-a)*quad(lambda mu: norm.pdf(p, 10**((10-mu)/5), .015), a, b,
                     epsabs=1e-12)[0] for a, b, m in zip(edges[:-1], edges[1:], mass))
        result = histogram_parallax(edges, mass, p, .015, order=64)
        np.testing.assert_allclose(result['log_score'], np.log(direct), atol=3e-6)
    ref = 12.; sigma = .3
    expected = sum(m/(b-a)*(norm.cdf((b-ref)/sigma)-norm.cdf((a-ref)/sigma))
                   for a, b, m in zip(edges[:-1], edges[1:], mass))
    np.testing.assert_allclose(histogram_reference_score(edges, mass, ref, sigma), np.log(expected), atol=1e-10)

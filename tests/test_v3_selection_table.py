import numpy as np
from test_v3_inference import model
from photdis.v3.selection import component_entry
from photdis.v3.selection_table import profile_table, SelectionTable
from photdis.v3.inference import distance_basis
from photdis.v3.likelihood import trapezoid_weights


def test_common_error_scale_is_exact_distance_translation():
    m = model(); ratio = np.array([1.3, 1., .7]); mu = np.linspace(5, 25, 101)
    a = component_entry(m, ratio*.4, mu, draws=128)
    b = component_entry(m, ratio, mu+2.5*np.log10(.4), draws=128)
    np.testing.assert_allclose(a, b, atol=1e-14)
    shift, alpha = profile_table(m, np.log([1.3, .7]), draws=128, step=.05)
    table = SelectionTable(np.log([[1.3, .7]]), shift, alpha[None])
    value, audit = table.evaluate((ratio*.4)[None])
    fine = np.arange(-15, 45.0001, .025)
    response = component_entry(m, ratio*.4, fine, draws=128)
    curve = .9*response[0]+.1*response[1]
    expected = (curve*trapezoid_weights(fine))@distance_basis(fine).T
    np.testing.assert_allclose(value[0], expected, atol=2e-5)
    assert audit['inside_profile_hull'][0]


def test_gaussian_survival_entry_matches_direct_multivariate_normal():
    from scipy.stats import multivariate_normal
    from photdis.v3.selection import gaussian_component_entry, OPTICAL
    from test_v3_inference import model
    m = model(); errors = np.array([.1, .2, .3]); grid = np.linspace(8, 18, 13)
    got = gaussian_component_entry(m, errors, grid, draws=4096)
    expected = np.zeros_like(got)
    for b, brightness in enumerate((1., m.binary_brightness_factor)):
        for k in range(len(m.weights)):
            for j, mu in enumerate(grid):
                a = brightness*10**(-.4*(mu-10))
                mean = a*m.means[k, OPTICAL]
                cov = a*a*m.covariances[k][np.ix_(OPTICAL, OPTICAL)]+np.diag(errors**2)
                expected[b, k, j] = multivariate_normal.cdf(mean-5*errors, mean=np.zeros(3), cov=cov,
                    maxpts=1000000, abseps=1e-8, releps=1e-8, rng=np.random.default_rng(1))
    np.testing.assert_allclose(got, expected, atol=2e-4)

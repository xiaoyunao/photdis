import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc, norm
from photdis.v3_1.candidate_selection import candidate_thresholds, selected_gaussian_probability, MAG_FACTOR


def test_thresholds_preserve_original_observed_brightness_and_error_cuts():
    error = np.array([.08, .15, .1]); trans = np.array([.8, .9, .95]); cal = np.array([.1, -.2, .05])
    eff, lower = candidate_thresholds(error, trans, cal)
    np.testing.assert_allclose(eff, error/trans*10**(.4*cal))
    raw_lower = lower*trans/10**(.4*cal)
    np.testing.assert_allclose(raw_lower, [MAG_FACTOR*error[0]/.1, 10., MAG_FACTOR*error[2]/.1])
    raw = np.array([[2., 12., 3.], [2., 9., 3.], [.5, 12., 3.]])
    selected = ((22.5-2.5*np.log10(raw[:, 1])) < 20) & ((MAG_FACTOR*error/raw).max(axis=1) < .1)
    np.testing.assert_array_equal(selected, (raw/trans*10**(.4*cal) > lower).all(axis=1))


def test_independent_gaussian_matches_analytic_survival_exactly():
    mean = np.array([[2., 3., 4.], [1., 2., 3.]])
    sigma = np.array([.5, 1., 1.5]); cov = np.broadcast_to(np.diag(sigma**2), (2, 3, 3))
    lower = np.array([1.5, 2.5, 3.5])
    actual = selected_gaussian_probability(mean, cov, lower, np.zeros(3), draws=32)
    np.testing.assert_allclose(actual, ndtr((mean-lower)/sigma).prod(axis=1), atol=1e-15)


def test_correlated_color_support_matches_independent_direct_qmc():
    mean = np.array([3., 4., 5.]); cov = np.array([[1., .4, .1], [.4, 1.5, .3], [.1, .3, 2.]])
    lower = np.array([1., 2., 3.]); cal = np.array([.2, -.1, .1])
    def support(m):
        return (m[:, 0]-m[:, 1] > .5) & (m[:, 1]-m[:, 2] < .25)
    estimate = selected_gaussian_probability(mean, cov, lower, cal, support=support, draws=2**16, seed=29)
    z = norm.ppf(qmc.Sobol(3, scramble=True, seed=37).random_base2(19))
    flux = mean+z@np.linalg.cholesky(cov).T; selected = (flux > lower).all(axis=1)
    accepted = np.zeros(len(flux), bool)
    accepted[selected] = support(22.5-2.5*np.log10(flux[selected])+cal)
    assert abs(estimate-accepted.mean()) < .001

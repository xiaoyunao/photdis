import numpy as np
from scipy.stats import multivariate_normal
from photdis.v3.evidence import bridge_integral


def test_bridge_recovers_known_multivariate_normal_normalization():
    mean = np.array([1., -2., .3]); covariance = np.array([[2., .5, 0], [.5, 1., .2], [0, .2, .6]])
    distribution = multivariate_normal(mean, covariance)
    rng = np.random.default_rng(941); samples = distribution.rvs(size=6000, random_state=rng)
    result = bridge_integral(lambda x: distribution.logpdf(x)+3.2, samples, proposal_draws=4096)
    assert result['fixed_point_converged']
    assert abs(result['log_integral']-3.2) < .02
    assert result['half_log_integral_difference'] < .03
    assert result['proposal_bridge_ess'] > 3000

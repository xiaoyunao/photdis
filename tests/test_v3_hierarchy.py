import numpy as np
from scipy.optimize import approx_fprime
from test_v3_inference import model
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.sampling import hmc, chain_diagnostics


def test_field_shrinkage_gradient_and_single_use_likelihood():
    rng = np.random.default_rng(23); m = model()
    r = rng.uniform(.01, 1, (15, 2, 3)); a = rng.uniform(.1, .9, r.shape)
    h = FieldPopulation(r, a, m, np.arange(15)%3)
    x = rng.normal(0, .2, h.dimension)
    value, gradient = h.objective(x)
    numerical = approx_fprime(x, lambda z: h.objective(z)[0], 1e-7)
    np.testing.assert_allclose(gradient, numerical, atol=2e-6)
    # Constant response/selection gives no population information, only the prior.
    flat = FieldPopulation(np.ones_like(r), np.ones_like(a), m, np.arange(15)%3)
    v, g = flat.objective(x)
    np.testing.assert_allclose(v, .5*(x@x), atol=1e-12)
    np.testing.assert_allclose(g, x, atol=1e-12)


def test_hmc_recovers_correlated_gaussian_and_chain_diagnostics_detect_stuck():
    mean = np.array([1., -2.]); covariance = np.array([[2., .7], [.7, 1.]])
    precision = np.linalg.inv(covariance)
    def objective(x):
        d = x-mean
        return .5*d@precision@d, precision@d
    records = [hmc(objective, mean, covariance, draws=1200, warmup=200, seed=i) for i in range(4)]
    samples = np.array([r['samples'] for r in records])
    np.testing.assert_allclose(samples.mean(axis=(0, 1)), mean, atol=.08)
    np.testing.assert_allclose(np.cov(samples.reshape(-1, 2).T), covariance, atol=.15)
    diagnostic = chain_diagnostics(samples)
    assert diagnostic['max_rhat'] < 1.03 and diagnostic['minimum_ess'] > 300
    samples[0, :, 0] += 4
    assert chain_diagnostics(samples)['max_rhat'] > 1.2


def test_hmc_rejects_out_of_domain_proposals():
    def objective(x):
        if abs(x[0]) > 1: raise ValueError('Outside posterior support')
        return .5*(x@x), x
    result = hmc(objective, np.zeros(1), np.eye(1), warmup=200, draws=500, seed=83)
    assert np.isfinite(result['samples']).all() and (abs(result['samples']) <= 1).all()
    assert abs(result['samples'].mean()) < .15

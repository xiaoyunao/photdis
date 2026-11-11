import numpy as np
from scipy.stats import norm
from test_v3_inference import model, observation
from photdis.v3.inference import infer_reference, target_posterior
from photdis.v3.compression import compress
from photdis.v3.target import component_response, joint_coefficients
from photdis.v3.marginal import marginal_coefficients, marginal_star, normal_ratio_maximum


def test_shared_posterior_marginal_is_average_of_normalized_native_posteriors():
    m = model(); o = observation(); p = infer_reference(m, o)['posterior']; record = compress(p, 2)
    response = component_response(record, m)[None]
    rng = np.random.default_rng(9); theta = rng.dirichlet(np.ones(15), 4).reshape(4, 3, 5)
    weights = np.array([.1, .2, .3, .4]); coef = joint_coefficients(m, theta, np.zeros((4, 2)))
    average, evidence = marginal_coefficients(response, coef, weights)
    got = marginal_star(record, m, average[0])
    expected = sum(w*target_posterior(m, o, p.mu, t)[0].distance_bins().reshape(2, 2, 3, 6).sum(axis=(0, 1)) for w, t in zip(weights, theta))
    np.testing.assert_allclose(got['class_distance'], expected, atol=5e-4)
    np.testing.assert_allclose(got['integral'], 1, atol=1e-12)
    assert evidence.shape == (1, 4) and got['omitted_posterior_bound'] < 1e-3


def test_prior_ratio_bound_covers_entire_real_line():
    x = np.linspace(-100, 100, 200001)
    for mean, sigma in [(8, 3), (11.5, 1.5), (14, 1.4), (16.5, 1.8), (20.5, 3.5)]:
        ratio = np.exp(norm.logpdf(x, mean, sigma)-norm.logpdf(x, 15, 5))
        assert ratio.max() <= normal_ratio_maximum(mean, sigma)*(1+1e-12)
        assert normal_ratio_maximum(mean, sigma)/ratio.max() < 1.00001

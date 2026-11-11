import numpy as np
from scipy.optimize import approx_fprime
from test_v3_inference import model, observation
from photdis.v3.inference import infer_reference, target_posterior
from photdis.v3.compression import compress
from photdis.v3.target import component_response, population_objective, joint_distance_probability


def test_population_sed_tilt_gradient_matches_finite_difference():
    rng = np.random.default_rng(91)
    response = rng.uniform(.01, 1, (12, 4, 2)); selection = rng.uniform(.1, .9, (12, 4, 2))
    weights = np.array([.1, .2, .3, .4]); classes = rng.dirichlet([2, 2, 1], size=4)
    features = rng.normal(size=(4, 2)); center = np.zeros(5); x = rng.normal(0, .2, 7)
    args = (response, selection, weights, classes, features, center)
    _, analytical = population_objective(x, *args)
    numerical = approx_fprime(x, lambda y: population_objective(y, *args)[0], epsilon=1e-7)
    np.testing.assert_allclose(analytical, numerical, atol=2e-6)


def test_component_response_recovers_direct_joint_population_posterior():
    m = model(); o = observation(); p = infer_reference(m, o)['posterior']
    compressed = compress(p, 2)
    response = component_response(compressed, m)
    theta = np.arange(1, 16, dtype=float).reshape(3, 5); theta /= theta.sum()
    got = joint_distance_probability(response[None], m, theta, np.zeros(2))[0]
    direct, _ = target_posterior(m, o, p.mu, theta)
    expected = direct.distance_bins().reshape(2, 2, 3, 6).sum(axis=(0, 1))
    np.testing.assert_allclose(got, expected, atol=5e-4)


def test_batched_joint_draws_equal_separate_exact_contractions():
    from photdis.v3.target import joint_coefficients, joint_draw_probabilities
    rng = np.random.default_rng(333); m = model()
    response = rng.uniform(.01, 1, (7, 2, 5, 6))
    theta = rng.dirichlet(np.ones(15), size=9).reshape(9, 3, 5)
    beta = rng.normal(0, .2, (9, 2))
    batched = joint_draw_probabilities(response, joint_coefficients(m, theta, beta))
    expected = np.array([joint_distance_probability(response, m, t, b) for t, b in zip(theta, beta)])
    np.testing.assert_allclose(batched, expected, atol=2e-15)

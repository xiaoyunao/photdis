import itertools
import numpy as np
import pytest
from photdis.v3.population import SharedCounts, selected_log_likelihood, joint_responsibilities


def test_selection_normalization_matches_analytic_two_component_case():
    response = np.array([[.2, .1], [.1, .3]])
    theta = np.array([[.25, .75], [.8, .2]])
    alpha = np.array([.4, .6])
    expected = np.array([sum(np.log(response@t))-2*np.log(t@alpha) for t in theta])
    np.testing.assert_allclose(selected_log_likelihood(np.log(response), theta, alpha), expected)
    p = joint_responsibilities(np.log(response), theta)
    np.testing.assert_allclose(p[0, 0], [.4, .6])


def test_shared_count_moments_match_exact_enumeration_and_streaming():
    p = np.array([[[.9, .1], [.8, .2]], [[.2, .8], [.1, .9]]])
    weights = np.array([.3, .7]); ids = np.array([1, 2])
    acc = SharedCounts(ids, 2, weights)
    acc.add(p[:, :1], ids); acc.add(p[:, 1:], ids)
    result = acc.result()
    values, probs = [], []
    for draw in range(2):
        for labels in itertools.product(range(2), repeat=2):
            values.append(np.bincount(labels, minlength=2))
            probs.append(weights[draw]*p[draw, 0, labels[0]]*p[draw, 1, labels[1]])
    values, probs = np.array(values), np.array(probs)
    mean = probs@values; residual = values-mean
    cov = (residual.T*probs)@residual
    np.testing.assert_allclose(result['mean'], mean)
    np.testing.assert_allclose(result['covariance'], cov)
    assert result['shared_covariance'][0, 0] > .4
    np.testing.assert_allclose(result['covariance'].sum(axis=0), 0, atol=1e-14)
    with pytest.raises(ValueError, match='indices differ'): acc.add(p, ids[::-1])


def test_large_counts_are_int64_and_final_mixture_precision_not_rare_response_max():
    ids = np.arange(4); p = np.zeros((4, 40000, 2)); p[:, :, 0] = 1
    acc = SharedCounts(ids, 2, seed=1)
    acc.add(p, ids); result = acc.result()
    assert result['count_draws'].dtype == np.int64
    assert (result['count_draws'][:, 0] == 40000).all()
    assert result['marginal_half_delta'] == 0
    # A tiny-weight opposite conditional is retained without dominating the final
    # marginalized precision diagnostic.
    p = np.array([[[1., 0]], [[1., 0]], [[0., 1]], [[1., 0]]])
    acc = SharedCounts(ids, 2, [1, 1, 1e-9, 1]); acc.add(p, ids)
    assert acc.result()['marginal_half_delta'] < 1e-8


def test_population_fit_respects_object_specific_entry_probabilities():
    from photdis.v3.population import fit_population, population_draws
    # Exactly informative observed class, but two ancillary-error groups have
    # different entry efficiencies. The underlying fraction is 0.4, not the
    # fraction of selected rows assigned to class zero.
    alpha = np.repeat([[.8, .2], [.3, .9]], [200, 200], axis=0)
    expected = [.4*.8/(.4*.8+.6*.2), .4*.3/(.4*.3+.6*.9)]
    labels = np.r_[np.r_[np.zeros(round(200*expected[0]), int), np.ones(200-round(200*expected[0]), int)],
                   np.r_[np.zeros(round(200*expected[1]), int), np.ones(200-round(200*expected[1]), int)]]
    lr = np.full((400, 2), -50.)
    lr[np.arange(400), labels] = 0
    fit = fit_population(lr, alpha, [.5, .5], prior_sigma=3.)
    assert abs(fit['weights'][0]-.4) < .01
    posterior = population_draws(fit, lr, alpha, draws=2048)
    assert posterior['effective_sample_size'] > 1000
    assert abs(posterior['posterior_weight']@posterior['weights'][:, 0]-.4) < .01

import numpy as np
import pytest
from photdis.v3.catalog_conditioning import CatalogConditioning
from photdis.v3.population import SharedCounts


def test_joint_catalog_update_matches_enumeration_and_shared_count_covariance():
    # Two global population states; two observed stars. Their common state is
    # inferred once from both observations, not independently per star.
    prior = np.array([.4, .6])
    likelihood = np.array([[.8, .2], [.1, .9]])
    alpha = np.array([[.9, .7], [.6, .8]])
    p = np.array([[[.9, .1], [.7, .3]], [[.2, .8], [.1, .9]]])
    accumulator = CatalogConditioning([11, 22], prior)
    accumulator.add(np.log(likelihood[:, :1]), alpha[:, :1], [False], [11, 22])
    accumulator.add(np.log(likelihood[:, 1:]), alpha[:, 1:], [False], [11, 22])
    result = accumulator.result()
    expected = prior*(likelihood/alpha).prod(axis=1)
    expected /= expected.sum()
    np.testing.assert_allclose(result['weights'], expected)
    count = SharedCounts([11, 22], 2, draw_weights=result['weights'])
    count.add(p, [11, 22])
    values, weights = [], []
    for state in range(2):
        for a in range(2):
            for b in range(2):
                values.append(np.bincount([a, b], minlength=2))
                weights.append(expected[state]*p[state, 0, a]*p[state, 1, b])
    values, weights = np.array(values), np.array(weights)
    mean = weights@values
    covariance = ((values-mean).T*weights)@(values-mean)
    np.testing.assert_allclose(count.result()['mean'], mean)
    np.testing.assert_allclose(count.result()['covariance'], covariance)
    assert abs(prior@p[:, :, 0].sum(axis=1)-mean[0]) > .01


def test_no_double_use_of_fitted_identity_and_order_guard():
    c = CatalogConditioning([3, 4], [.5, .5])
    c.add(np.log([[1e-20, .2], [.9, .8]]), np.ones((2, 2)), [True, False], [3, 4])
    r = c.result()
    np.testing.assert_allclose(r['weights'], [.2, .8])
    assert r['new_objects'] == r['reused_objects'] == 1
    with pytest.raises(ValueError):
        c.add(np.zeros((2, 1)), np.ones((2, 1)), [False], [4, 3])


def test_concentrated_importance_weights_are_exposed_not_accepted():
    c = CatalogConditioning(np.arange(100), np.ones(100))
    c.add(np.arange(100)[:, None]*100., np.ones((100, 1)), [False], np.arange(100))
    assert c.result()['effective_sample_size'] == 1
    assert c.result()['maximum_weight'] == 1

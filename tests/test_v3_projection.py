import numpy as np
import v3_0_projection as projection


def test_field_namespaces_keep_shared_parameters_but_independent_latent_noise():
    ids = np.arange(2048)
    probabilities = np.tile([[[.4, .6], [.7, .3]]], (len(ids), 1, 1))
    results = []
    for namespace in (11, 12, 11):
        projection._LATENT_SEED = namespace
        counts = projection.ProjectionCounts(ids, 2, seed=308000)
        counts.add(probabilities, ids)
        results.append(counts.result())
    np.testing.assert_array_equal(results[0]['draw_ids'], results[1]['draw_ids'])
    np.testing.assert_allclose(results[0]['covariance'], results[1]['covariance'])
    np.testing.assert_array_equal(results[0]['count_draws'], results[2]['count_draws'])
    assert not np.array_equal(results[0]['count_draws'], results[1]['count_draws'])

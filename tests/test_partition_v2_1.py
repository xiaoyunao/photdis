import numpy as np
from photdis.baseline_v0 import HistogramQRF
from photdis.partition_v2_1 import class_partition_candidate


def test_partition_candidate_preserves_members_and_does_not_modify_control():
    rng = np.random.default_rng(14)
    m = rng.normal(size=(300, 5))
    y = rng.normal(size=300)
    labels = np.tile(np.arange(3), 100)
    control = HistogramQRF(3, n_estimators=4, min_samples_leaf=10).fit(m, y, labels)
    before = control.distribution(control.feature_matrix(m[:10]))[0].copy()
    candidate = class_partition_candidate(control, m, y, labels)
    actual = candidate.distribution(candidate.feature_matrix(m))
    np.testing.assert_allclose(actual[0].sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(actual[1].sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(actual[1][:, 3], 0)
    np.testing.assert_array_equal(before, control.distribution(control.feature_matrix(m[:10]))[0])
    for key in ['n_estimators', 'min_samples_leaf', 'max_depth', 'max_features', 'bootstrap', 'random_state']:
        assert control.forest.get_params()[key] == candidate.forest.get_params()[key]

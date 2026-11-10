import numpy as np
from photdis.information_v2_1 import fit_calibration, calibrated, class_score, local_neighbors


def test_calibration_uses_common_probability_contract():
    labels = np.tile(np.arange(3), 20)
    p = np.full((60, 3), .1)
    p[np.arange(60), labels] = .8
    theta = fit_calibration(p, labels)
    q = calibrated(p, theta)
    assert np.allclose(q.sum(axis=1), 1)
    assert class_score(q, labels, np.full_like(q, 1/3))['log_gain_over_prior'] > 0


def test_local_neighbors_includes_shared_band_covariance():
    train = np.array([[10., 10., 10.], [10.02, 10., 9.98], [10., 10.04, 10.]])
    result = local_neighbors(train, np.full_like(train, .02), np.array([0, 1, 1]),
                             np.array([5., -2., -3.]), train[:1], np.full((1, 3), .02), np.array([0]))
    assert result['neighbors'][0] == 3
    assert result['luminosity_span'][0] == 8
    assert 0 < result['same_class_mass'][0] < 1

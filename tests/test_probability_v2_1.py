import numpy as np
from photdis.baseline_v0 import HistogramQRF
from photdis.baseline_v1 import integrate, shift_pdf
from photdis.probability_v2_1 import (
    JointReferenceForest, shift_joint, near_mass, migration, color_covariance, psf_admission)


def test_joint_reproduces_frozen_model_and_preserves_class_distance_dependence():
    rng = np.random.default_rng(411)
    mags = rng.normal(16, 1, (240, 5))
    labels = np.tile([-1, 0, 1, 2], 60)
    y = np.where(labels == 1, -2., 7.) + rng.normal(0, .2, len(mags))
    model = HistogramQRF(2, n_estimators=5, min_samples_leaf=12).fit(mags, y, labels)
    joint = JointReferenceForest(model, mags, y, labels, np.tile([1, 2], 120))
    m, e, ids = mags[:17], np.full((17, 5), .12), np.arange(17)
    actual = joint.predict(m, e, ids, draws=8)
    expected = integrate(model, m, e, ids, draws=8)['wide']
    np.testing.assert_allclose(actual['pdf'], expected['pdf'], atol=2e-6)
    np.testing.assert_allclose(actual['mu'], expected['mu'], atol=2e-5)
    np.testing.assert_allclose(actual['joint'][:, :4].sum(axis=2), expected['weights'], atol=2e-6)
    np.testing.assert_allclose(actual['joint'][:, 4:].sum(axis=1), expected['pdf'], atol=2e-6)
    q = near_mass(actual['pdf'], model.edges, m[:, 0])
    giant_near = near_mass(actual['joint'][:, 1], model.edges, m[:, 0])
    assert np.max(abs(giant_near-q*expected['weights'][:, 1])) > .02


def test_joint_transport_extremes_and_no_gaussian_near_mass():
    joint = np.array([[[.25, 0, 0, 0], [0, 0, 0, .75]]])
    edges = np.arange(5.)
    for delta in [-99., -.3, 0., .4, 99.]:
        shifted = shift_joint(joint, np.array([delta]), edges)
        np.testing.assert_allclose(shifted.sum(axis=2), [[.25, .75]])
        np.testing.assert_allclose(shifted.sum(axis=1), shift_pdf(joint.sum(axis=1), np.array([delta]), edges))
    np.testing.assert_allclose(near_mass(joint.sum(axis=1), edges, [17.]), [.75])


def test_migration_psf_and_covariance():
    result = migration([14, 15, 16, np.nan], [True, True, False, True], [.1, .2, .3, .1])
    assert result['n'] == 3 and result['far_contamination'] == .5
    assert result['boundary_crossing'] == 1
    keep, counts = psf_admission([' PSF ', 'REX', '', 'psf'])
    assert keep.tolist() == [True, False, False, True]
    assert counts == dict(total=4, admitted=2, non_psf=1, missing_type=1)
    np.testing.assert_allclose(color_covariance([[1, 2, 3]])[0], [[5, -4], [-4, 13]])


def test_large_leaf_accumulates_counts_before_float_normalization():
    from types import SimpleNamespace
    n = 50000
    tree = SimpleNamespace(apply=lambda x: np.zeros(len(x), dtype=int))
    model = SimpleNamespace(feature_matrix=lambda x: x, edges=np.array([0., 1.]), centers=np.array([.5]),
                            forest=SimpleNamespace(estimators_=[tree]), maps=[np.array([0])],
                            tables=[np.ones((1, 1), dtype='f4')], types=[np.array([[1., 0, 0, 0]], dtype='f4')])
    joint = JointReferenceForest(model, np.zeros((n, 2)), np.full(n, .5),
                                  np.zeros(n, int), np.ones(n, int))
    np.testing.assert_array_equal(joint.reconstruction_max_errors, [0., 0., 0.])

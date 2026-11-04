import pickle
import numpy as np
import pytest
from photdis.optical_v1_4 import optical_noise, OpticalQRF
from photdis.reference_v1_4 import optical_features
from photdis.baseline_v1 import qmc_noise


def test_noise_identity_chunk_invariance_and_shared_colors():
    ids = np.array([7, 91, -11010000000000001])
    z = optical_noise(ids, 17)
    assert np.array_equal(z, np.concatenate([optical_noise(ids[:1], 17), optical_noise(ids[1:], 17)]))
    assert np.array_equal(z[::-1], optical_noise(ids[::-1], 17))
    m = np.array([[20., 19., 18.5, 18.]]) + z * .1
    a, b = optical_features(m, 'grz'), optical_features(m, 'griz')
    assert np.array_equal(a[:, 0], b[:, 0])
    assert np.allclose(a[:, 1], b[:, 1] + b[:, 2])
    assert qmc_noise(ids, 17).shape == (3, 5)


def test_sobol_color_covariance_matches_shared_band_errors():
    errors = np.array([.02, .03, .04, .05])
    noise = np.concatenate([optical_noise(np.array([74]), d) for d in range(1024)]) * errors
    colors = optical_features(noise, 'griz')
    # Cov(g-r,r-i)=-Var(r); Cov(r-i,i-z)=-Var(i).
    expected = np.diag(errors[:-1]**2 + errors[1:]**2)
    expected[0, 1] = expected[1, 0] = -errors[1]**2
    expected[1, 2] = expected[2, 1] = -errors[2]**2
    assert np.allclose(np.cov(colors.T), expected, atol=3e-5)


def test_optical_fit_prediction_roundtrip_and_no_row_order_effect():
    rng = np.random.default_rng(84)
    mags = rng.normal(size=(64, 4)) + [20, 19, 18.5, 18]
    y = 3 * (mags[:, 0] - mags[:, 1]) + .1 * rng.normal(size=64)
    labels = np.arange(64) % 3
    for variant in ('grz', 'griz'):
        model = OpticalQRF(variant, n_estimators=3, min_samples_leaf=4, max_depth=3)
        model.fit(mags, y, labels)
        saved = pickle.loads(pickle.dumps(model))
        assert saved.band_order == ('G', 'R', 'I', 'Z')
        ids, errors = np.arange(5), np.full((5, 4), .02)
        p = model.predict(mags[:5], errors, ids, draws=4)
        q = saved.predict(mags[:5][::-1], errors, ids[::-1], draws=4)
        assert np.allclose(p['pdf'].sum(axis=1), 1)
        assert np.all(p['pdf'] >= 0)
        assert np.all(np.diff(p['q_Meffective'], axis=1) >= 0)
        assert np.array_equal(p['pdf'], q['pdf'][::-1])
        with pytest.raises(ValueError):
            saved.predict(mags[:5], np.full((5, 4), np.nan), ids)


def test_classifier_matches_fixed_forest_and_preserves_class_semantics():
    from photdis.optical_v1_4 import OpticalClassifier
    from photdis.baseline_v1_2 import DirectClassifier, balanced_weights
    rng = np.random.default_rng(91)
    mags = rng.normal(size=(60, 4)) + [20, 19, 18.5, 18]
    labels = np.array([0] * 30 + [1] * 10 + [2] * 20)
    forest = dict(n_estimators=3, min_samples_leaf=3, max_depth=4, max_features=1.0)
    for variant, dim in [('grz', 2), ('griz', 3)]:
        model = OpticalClassifier(variant, **forest).fit(mags, labels)
        selected = mags[:, [0, 1, 3]] if variant == 'grz' else mags
        control = DirectClassifier(dim, 'forest', dict(seed=20261103, forest=forest))
        control.fit(selected, labels, balanced_weights(labels))
        assert np.allclose(model.predict_proba(mags), control.predict_proba(selected), atol=1e-14)
        assert model.training_class_counts.tolist() == [30, 10, 20]
        with pytest.raises(ValueError, match='unknown is not negative'):
            model.fit(mags, np.where(labels == 1, -1, labels))
        with pytest.raises(ValueError, match='All three classes'):
            model.fit(mags, np.where(labels == 1, 0, labels))


def test_classifier_marginalization_is_serializable_and_i_blind_for_grz():
    from photdis.optical_v1_4 import OpticalClassifier
    rng = np.random.default_rng(94)
    mags = rng.normal(size=(60, 4)) + [20, 19, 18.5, 18]
    labels = np.arange(60) % 3
    model = OpticalClassifier('grz', n_estimators=3, min_samples_leaf=3, max_depth=3).fit(mags, labels)
    saved = pickle.loads(pickle.dumps(model))
    ids, errors = np.arange(5), np.full((5, 4), .05)
    p = model.predict_marginalized(mags[:5], errors, ids, draws=4)
    q = saved.predict_marginalized(mags[:5][::-1], errors, ids[::-1], draws=4)
    assert np.allclose(p, q[::-1], atol=1e-14)
    assert np.allclose(p.sum(axis=1), 1) and np.all(p >= 0)
    altered = mags[:5].copy()
    altered[:, 2] += 100
    errors[:, 2] = 10
    assert np.allclose(p, saved.predict_marginalized(altered, errors, ids, draws=4), atol=1e-14)
    zero = saved.predict_marginalized(mags[:5], np.zeros_like(errors), ids, draws=4)
    assert np.allclose(zero, saved.predict_proba(mags[:5]), atol=1e-14)
    assert saved.band_order == ('G', 'R', 'I', 'Z') and saved.class_order == (0, 1, 2)

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

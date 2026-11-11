import copy
import numpy as np
from photdis.v3.sed import EmpiricalSED
from photdis.v3.calibration import reference_nodes
from photdis.v3.uncertainty import reference_logratio, reference_quadrature_components, shared_case_weights, one_step_ensemble


def test_source_quadrature_reweighting_changes_all_calibration_parameters():
    channel = np.array([1]); distance = np.array([2.]); sigma = np.array([.2])
    base = np.array([[.01, .1, .02], [-.01, .2, .03]])
    mu, w = reference_nodes(channel, distance, sigma, [1.], [.1],
        offset=.01, extra=.1, outlier=.02, nodes=128)
    draws = np.array([base.copy(), base.copy()]); draws[1, 0] = [.04, .13, .03]
    broad = reference_quadrature_components(mu, channel, 10+5*np.log10(distance), sigma, base, repeats=1)
    logratio, normalization = reference_logratio(mu, channel, 10+5*np.log10(distance),
        sigma, base, draws, w, broad)
    np.testing.assert_allclose(normalization[0], 0, atol=1e-13)
    p = np.exp(logratio[1])*w
    np.testing.assert_allclose((p*mu).sum(), 10+5*np.log10(2)+.04, atol=2e-5)
    expected_variance = .97*(.2**2+.13**2)+.03*(.2**2+2.**2)
    np.testing.assert_allclose((p*(mu-(10+5*np.log10(2)+.04))**2).sum(), expected_variance, atol=2e-5)


def test_zero_perturbation_ensemble_equals_one_actual_em_update():
    rng = np.random.default_rng(73); n = 120
    y = rng.normal(size=(n, 1, 7))+np.array([6]*6+[0])
    noise = np.broadcast_to(np.eye(7)*.1, (n, 1, 7, 7))
    mask = np.ones((n, 7), bool); labels = np.zeros(n, int); labels[::3] = -1
    logw = np.zeros((n, 1)); features = y[:, 0, :3]
    m = EmpiricalSED(1, max_iter=1).fit(y, noise, mask, logw, labels, features)
    expected = copy.deepcopy(m); expected.max_iter = 2
    expected.fit(y, noise, mask, logw, labels, features, resume=True)
    cloud = dict(y=y, noise=noise, mask=mask, labels=labels, logweight=logw)
    draws, _ = one_step_ensemble(m, cloud, np.zeros((2, n, 1)), np.ones((2, n)))
    for actual in draws:
        np.testing.assert_allclose(actual.means, expected.means, atol=1e-12)
        np.testing.assert_allclose(actual.covariances, expected.covariances, atol=1e-12)
        np.testing.assert_allclose(actual.classes, expected.classes, atol=1e-12)
    weights, cells, sky = shared_case_weights(np.arange(n), np.arange(n)//30, draws=32)
    assert weights.shape == (32, n) and sky.shape == (32, 4)
    np.testing.assert_allclose(weights.sum(axis=1), n)

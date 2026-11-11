import numpy as np
from scipy.stats import multivariate_normal
from photdis.v3.sed import projected_conditionals, EmpiricalSED


def test_projected_conditionals_match_independent_joint_gaussian_conditioning():
    mean = np.array([2., 3., -.8]); v = np.array([[2, .3, .2], [.3, 1., -.1], [.2, -.1, .5]])
    y = np.array([[1., np.nan, -1.]])
    noise = np.diag([.2, 0., .1])[None]
    b, B, ll = projected_conditionals(y, noise, [1, 0, 1], mean, v)
    R = np.array([[1., 0, 0], [0, 0, 1.]])
    S = np.diag([.2, .1]); observed = np.array([1., -1.])
    total = R@v@R.T+S
    gain = v@R.T@np.linalg.inv(total)
    np.testing.assert_allclose(b[0], mean+gain@(observed-R@mean))
    np.testing.assert_allclose(B[0], v-gain@R@v)
    np.testing.assert_allclose(ll[0], multivariate_normal.logpdf(observed, R@mean, total))


def test_missing_label_not_a_physical_class_and_error_deconvolution():
    rng = np.random.default_rng(41); n = 600
    latent_mean = np.array([5., 6., 4., 4.5, 3., 2., -.7])
    v = np.diag([.3]*6+[.1]); s = np.diag([.2]*6+[.05])
    y = rng.multivariate_normal(latent_mean, v+s, n)[:, None, :]
    noise = np.broadcast_to(s, (n, 1, 7, 7)).copy(); mask = np.ones((n, 7), bool)
    mask[:100, 4] = False; y[:100, 0, 4] = np.nan
    labels = np.zeros(n, int); labels[::2] = -1
    model = EmpiricalSED(components=1, max_iter=12).fit(y, noise, mask, np.zeros((n, 1)), labels, y[:, 0, :3])
    assert model.classes.shape == (1, 3) and model.classes[0, 0] > .99
    np.testing.assert_allclose(model.means[0], latent_mean, atol=.12)
    np.testing.assert_allclose(np.diag(model.covariances[0]), np.diag(v), atol=.07)

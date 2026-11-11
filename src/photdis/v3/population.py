"""Selection-aware mixture inference and streaming, shared-parameter counts."""
import numpy as np
from scipy.special import logsumexp


def selected_log_likelihood(log_response, weights, selection_probability, object_weights=None):
    """Unselected component response p(data|component), normalized by P(entry).

    Components may be branch x continuous-distance basis integrals. Do not feed
    a previously selected-normalized response and divide by selection again.
    """
    lr = np.asarray(log_response, float)
    theta = np.atleast_2d(weights).astype(float)
    alpha = np.asarray(selection_probability, float)
    if (theta < 0).any() or not np.allclose(theta.sum(axis=1), 1):
        raise ValueError('Normalized nonnegative mixture weights required')
    if (alpha < 0).any() or (alpha > 1).any() or not np.isfinite(alpha).all():
        raise ValueError('Invalid selection probabilities')
    w = np.ones(len(lr)) if object_weights is None else np.asarray(object_weights, float)
    if (w < 0).any() or not np.isfinite(w).all(): raise ValueError('Invalid sampling weights')
    with np.errstate(divide='ignore'):
        logtheta = np.log(theta)
    norm = theta@alpha
    if (norm <= 0).any(): raise ValueError('Zero selected population probability')
    return logsumexp(lr[None, :, :]+logtheta[:, None, :], axis=-1)@w-w.sum()*np.log(norm)


def joint_responsibilities(log_response, weights):
    theta = np.atleast_2d(weights)
    if (theta < 0).any() or not np.allclose(theta.sum(axis=1), 1):
        raise ValueError('Normalized mixture weights required')
    with np.errstate(divide='ignore'):
        logp = np.asarray(log_response)[None, :, :]+np.log(theta)[:, None, :]
    norm = logsumexp(logp, axis=-1, keepdims=True)
    if not np.isfinite(norm).all(): raise ValueError('No supported component')
    return np.exp(logp-norm)


class SharedCounts:
    """Streaming exact first/second moments, conditional on shared global draws.

    Chunks must pass identical draw_ids in identical order. Full-catalog counts
    use unit weights. Sample-weighted totals are intentionally not accepted here:
    their design variance belongs to an explicit sampling-design estimator.
    """
    def __init__(self, draw_ids, n_components, draw_weights=None, seed=None):
        self.draw_ids = np.asarray(draw_ids, 'i8')
        if len(np.unique(self.draw_ids)) != len(self.draw_ids) or len(self.draw_ids) < 2:
            raise ValueError('At least two unique shared draws required')
        s = len(self.draw_ids)
        w = np.ones(s) if draw_weights is None else np.asarray(draw_weights, float)
        if len(w) != s or (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('Invalid shared posterior draw weights')
        self.weights = w/w.sum()
        self.conditional_mean = np.zeros((s, n_components), 'f8')
        self.mean_conditional_cov = np.zeros((n_components, n_components), 'f8')
        self.realized = None if seed is None else np.zeros((s, n_components), 'i8')
        self.rng = None if seed is None else np.random.default_rng(seed)
        self.n = 0; self.marginal_half_delta = 0.

    def add(self, probabilities, draw_ids):
        if not np.array_equal(draw_ids, self.draw_ids):
            raise ValueError('Chunk global-draw indices differ')
        p = np.asarray(probabilities, float)
        if p.ndim != 3 or p.shape[0] != len(self.draw_ids) or p.shape[2] != self.conditional_mean.shape[1]:
            raise ValueError('Expected shared_draw x object x component')
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=2), 1, atol=1e-10):
            raise ValueError('Invalid per-object joint probabilities')
        means = p.sum(axis=1); self.conditional_mean += means
        self.mean_conditional_cov += np.diag(self.weights@means)
        self.mean_conditional_cov -= np.einsum('s,sik,sil->kl', self.weights, p, p, optimize=True)
        # Precision is on final response/calibration-marginal joint probabilities.
        # Interleaved halves share the same globally weighted mixture, with their
        # own normalizations; never max over response-specific conditional errors.
        halves = []
        for select in (np.arange(len(p)) % 2 == 0, np.arange(len(p)) % 2 == 1):
            w = self.weights[select]
            if w.sum() == 0: halves = []; break
            halves.append(np.einsum('s,sik->ik', w/w.sum(), p[select]))
        delta = float(abs(halves[0]-halves[1]).max()) if halves else np.inf
        self.marginal_half_delta = max(self.marginal_half_delta, delta)
        if self.realized is not None:
            cdf = np.cumsum(p, axis=2); cdf[:, :, -1] = 1.
            labels = (self.rng.random(p.shape[:2])[..., None] > cdf).sum(axis=2)
            for k in range(p.shape[2]): self.realized[:, k] += (labels == k).sum(axis=1, dtype='i8')
        self.n += p.shape[1]
        return np.einsum('s,sik->ik', self.weights, p)

    def result(self):
        mean = self.weights@self.conditional_mean
        delta = self.conditional_mean-mean
        shared = (delta.T*self.weights)@delta
        covariance = self.mean_conditional_cov+shared
        # Population posterior covariance uses normalized quadrature expectation,
        # not the unbiased finite-sample covariance correction.
        result = dict(n=self.n, mean=mean, covariance=covariance,
                      conditional_covariance=self.mean_conditional_cov.copy(),
                      shared_covariance=shared, conditional_means=self.conditional_mean.copy(),
                      draw_ids=self.draw_ids.copy(), draw_weights=self.weights.copy(),
                      marginal_half_delta=self.marginal_half_delta,
                      draw_effective_sample_size=float(1/(self.weights@self.weights)))
        if self.realized is not None:
            assert self.realized.dtype == np.int64 and (self.realized.sum(axis=1) == self.n).all()
            result['count_draws'] = self.realized.copy()
        return result

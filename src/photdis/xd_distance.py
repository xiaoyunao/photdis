"""Error-deconvolved color--luminosity mixtures and correlated distance inference.

The latent vector is (Mg, g-r, r-z[, z-W1[, W1-W2]]).  Errors in repeated
bands are propagated from band magnitudes, not treated as independent colors.
The selected training population is the fitted population; its mixture weights
are not Galactic population fractions.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr
from sklearn.mixture import GaussianMixture


def measurement_covariance(band_sigma, parallax_fractional_sigma=None):
    """Covariance of (Mg or g0, adjacent-band colors) from independent bands.

    The optional fractional parallax error adds only to the training Mg term.
    Shared g/r/z/W1 errors induce all relevant off-diagonal covariance terms.
    Extinction-map and interband calibration covariance are not supplied by
    the catalogs and are outside this low-extinction approximation.
    """
    sigma = np.asarray(band_sigma, float)
    if sigma.ndim != 2 or sigma.shape[1] not in (3, 4, 5):
        raise ValueError("Expected rows of g/r/z[/W1[/W2]] errors")
    if not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("Band errors must be finite and nonnegative")
    n, bands = sigma.shape
    transform = np.zeros((bands, bands), float)
    transform[0, 0] = 1
    for j in range(1, bands):
        transform[j, j - 1] = 1
        transform[j, j] = -1
    cov = np.einsum("ab,nb,cb->nac", transform, sigma**2, transform)
    if parallax_fractional_sigma is not None:
        fraction = np.asarray(parallax_fractional_sigma, float)
        if fraction.shape != (n,) or not np.isfinite(fraction).all() or np.any(fraction < 0):
            raise ValueError("Invalid fractional parallax error")
        cov[:, 0, 0] += (5 / np.log(10) * fraction) ** 2
    return cov


def _positive_covariance(covariance, floor):
    sym = (covariance + np.swapaxes(covariance, -1, -2)) * .5
    value, vector = np.linalg.eigh(sym)
    value = np.maximum(value, floor)
    return np.einsum("...ij,...j,...kj->...ik", vector, value, vector)


class XDMixture:
    """Full-covariance Gaussian mixture fitted with heteroscedastic XD EM."""

    def __init__(self, components, *, covariance_floor=1e-4, max_iter=40,
                 tolerance=1e-4, seed=20261029, batch_size=256):
        if components < 1 or covariance_floor <= 0 or max_iter < 1 or batch_size < 1:
            raise ValueError("Invalid mixture configuration")
        self.components = int(components)
        self.covariance_floor = float(covariance_floor)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.seed = int(seed)
        self.batch_size = int(batch_size)

    def fit(self, observed, errors):
        x = np.asarray(observed, float)
        s = np.asarray(errors, float)
        if x.ndim != 2 or s.shape != (len(x), x.shape[1], x.shape[1]):
            raise ValueError("Observed values and covariance shape differ")
        if len(x) < self.components * 4 or not np.isfinite(x).all() or not np.isfinite(s).all():
            raise ValueError("Insufficient or nonfinite training rows")
        np.linalg.cholesky(s + np.eye(x.shape[1]) * 1e-12)
        initial = GaussianMixture(n_components=self.components,
                    covariance_type="full", reg_covar=self.covariance_floor,
                    n_init=1, random_state=self.seed, max_iter=100).fit(x)
        self.weights_ = initial.weights_.copy()
        self.means_ = initial.means_.copy()
        self.covariances_ = _positive_covariance(
            initial.covariances_ - np.mean(s, axis=0)[None, :, :],
            self.covariance_floor)
        self.log_likelihood_history_ = []
        self.converged_ = False
        n, d = x.shape
        for iteration in range(self.max_iter):
            count = np.zeros(self.components)
            first = np.zeros((self.components, d))
            second = np.zeros((self.components, d, d))
            log_like = 0.0
            for start in range(0, n, self.batch_size):
                xb = x[start:start+self.batch_size]
                sb = s[start:start+self.batch_size]
                cov = self.covariances_[None, :, :, :] + sb[:, None, :, :]
                chol = np.linalg.cholesky(cov)
                delta = xb[:, None, :] - self.means_[None, :, :]
                whitened = np.linalg.solve(chol, delta[..., None])[..., 0]
                logdet = 2 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
                log_component = (np.log(self.weights_)[None, :] - .5 *
                    (d * np.log(2*np.pi) + logdet + np.sum(whitened**2, axis=-1)))
                normalizer = logsumexp(log_component, axis=1)
                resp = np.exp(log_component - normalizer[:, None])
                log_like += normalizer.sum()
                precision_delta = np.linalg.solve(
                    np.swapaxes(chol, -1, -2), whitened[..., None])[..., 0]
                latent_mean = self.means_[None, :, :] + np.einsum(
                    "kde,bke->bkd", self.covariances_, precision_delta)
                right = np.broadcast_to(self.covariances_, cov.shape)
                inverse_right = np.linalg.solve(np.swapaxes(chol, -1, -2),
                    np.linalg.solve(chol, right))
                latent_cov = self.covariances_[None] - np.einsum(
                    "kde,bkef->bkdf", self.covariances_, inverse_right)
                count += resp.sum(axis=0)
                first += np.einsum("bk,bkd->kd", resp, latent_mean)
                second += np.einsum("bk,bkde->kde", resp, latent_cov)
                second += np.einsum("bk,bkd,bke->kde", resp, latent_mean, latent_mean)
            self.log_likelihood_history_.append(float(log_like))
            change = log_like - self.log_likelihood_history_[-2] if iteration else np.inf
            if iteration >= 5 and 0 <= change < self.tolerance * max(1, abs(log_like)):
                self.converged_ = True
                break
            safe = np.maximum(count, 1e-12)
            means = first / safe[:, None]
            covariance = second / safe[:, None, None] - np.einsum(
                "kd,ke->kde", means, means)
            self.weights_ = np.maximum(count / n, 1e-12)
            self.weights_ /= self.weights_.sum()
            self.means_ = means
            self.covariances_ = _positive_covariance(covariance,
                                                     self.covariance_floor)
        return self

    def conditional_mg_logpdf(self, observed, errors):
        """Held-out log p(observed Mg | observed colors), with known errors."""
        x = np.asarray(observed, float)
        s = np.asarray(errors, float)
        if x.ndim != 2 or s.shape != (len(x), x.shape[1], x.shape[1]):
            raise ValueError("Observed values and covariance shape differ")
        output = np.empty(len(x), float)
        for start in range(0, len(x), self.batch_size):
            xb = x[start:start+self.batch_size]
            sb = s[start:start+self.batch_size]
            values = []
            for offset in (0, 1):
                y = xb[:, offset:]
                mean = self.means_[:, offset:]
                cov = (self.covariances_[None, :, offset:, offset:] +
                       sb[:, None, offset:, offset:])
                chol = np.linalg.cholesky(cov)
                delta = y[:, None, :] - mean[None]
                white = np.linalg.solve(chol, delta[..., None])[..., 0]
                logdet = 2*np.log(np.diagonal(chol, axis1=-2,
                                              axis2=-1)).sum(axis=-1)
                terms = np.log(self.weights_)[None] - .5*(
                    y.shape[1]*np.log(2*np.pi) + logdet +
                    np.sum(white**2, axis=-1))
                values.append(logsumexp(terms, axis=1))
            output[start:start+len(xb)] = values[0]-values[1]
        return output

    def posterior_mu(self, observed, errors, *, mu_min=0., mu_max=20.,
                     quantiles=(.025, .16, .5, .84, .975)):
        """Quantiles under a fixed flat prior in distance modulus [mu_min,mu_max]."""
        x = np.asarray(observed, float)
        s = np.asarray(errors, float)
        if x.ndim != 2 or s.shape != (len(x), x.shape[1], x.shape[1]):
            raise ValueError("Observed values and covariance shape differ")
        if mu_min >= mu_max:
            raise ValueError("Invalid distance-modulus prior")
        q = np.asarray(quantiles, float)
        if np.any((q <= 0) | (q >= 1)):
            raise ValueError("Quantiles must lie in (0,1)")
        result = np.empty((len(x), len(q)), float)
        for start in range(0, len(x), self.batch_size):
            xb = x[start:start+self.batch_size]
            sb = s[start:start+self.batch_size]
            cov = self.covariances_[None, :, :, :] + sb[:, None, :, :]
            chol = np.linalg.cholesky(cov)
            delta = xb[:, None, :] - self.means_[None, :, :]
            white = np.linalg.solve(chol, delta[..., None])[..., 0]
            inv_delta = np.linalg.solve(np.swapaxes(chol, -1, -2),
                                        white[..., None])[..., 0]
            inverse_axis = np.linalg.solve(np.swapaxes(chol, -1, -2),
                np.linalg.solve(chol, np.eye(x.shape[1])[0][None, None, :, None]))[..., 0]
            a = inverse_axis[..., 0]
            mean_mu = inv_delta[..., 0] / a
            sd_mu = np.sqrt(1 / a)
            lo = ndtr((mu_min - mean_mu) / sd_mu)
            hi = ndtr((mu_max - mean_mu) / sd_mu)
            mass = np.maximum(hi-lo, 1e-300)
            logdet = 2 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
            chi2 = np.sum(white**2, axis=-1) - inv_delta[..., 0]**2/a
            log_weight = (np.log(self.weights_)[None] - .5*logdet - .5*chi2
                          - .5*np.log(a) + np.log(mass))
            weight = np.exp(log_weight - logsumexp(log_weight, axis=1)[:, None])
            for j, probability in enumerate(q):
                left = np.full(len(xb), mu_min)
                right = np.full(len(xb), mu_max)
                for _ in range(38):
                    mid = (left+right)/2
                    component_cdf = (ndtr((mid[:, None]-mean_mu)/sd_mu)-lo)/mass
                    cdf = np.einsum("bk,bk->b", weight,
                                    np.clip(component_cdf, 0, 1))
                    right = np.where(cdf >= probability, mid, right)
                    left = np.where(cdf < probability, mid, left)
                result[start:start+len(xb), j] = (left+right)/2
        return result

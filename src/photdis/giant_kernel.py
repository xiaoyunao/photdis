"""Conditional Legacy-g luminosity distribution for a spectroscopic-giant sample.

The input is only dereddened Legacy grz color. Catalog luminosity errors enter
as uncertainty on each kernel component, and shared r-band color errors are
retained in the two-dimensional color covariance.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr


def color_covariance(sigma_g, sigma_r, sigma_z):
    """Covariance of (g-r, r-z) for independent band measurements."""
    sg, sr, sz = (np.asarray(value, float) for value in
                  (sigma_g, sigma_r, sigma_z))
    out = np.empty(np.broadcast_shapes(sg.shape, sr.shape, sz.shape) + (2, 2))
    out[..., 0, 0] = sg * sg + sr * sr
    out[..., 0, 1] = out[..., 1, 0] = -sr * sr
    out[..., 1, 1] = sr * sr + sz * sz
    return out


def conditional_weights(train_colors, train_covariances, query_color,
                        query_covariance, bandwidth):
    """Normalized color-kernel weights and their effective sample size."""
    colors = np.asarray(train_colors, float)
    covariance = (np.asarray(train_covariances, float)
                  + np.asarray(query_covariance, float)
                  + float(bandwidth) ** 2 * np.eye(2))
    diff = colors - np.asarray(query_color, float)
    determinant = (covariance[:, 0, 0] * covariance[:, 1, 1]
                   - covariance[:, 0, 1] ** 2)
    if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0):
        raise ValueError("Invalid color covariance")
    quadratic = (covariance[:, 1, 1] * diff[:, 0] ** 2
                 - 2 * covariance[:, 0, 1] * diff[:, 0] * diff[:, 1]
                 + covariance[:, 0, 0] * diff[:, 1] ** 2) / determinant
    log_weight = -.5 * (quadratic + np.log(determinant))
    weight = np.exp(log_weight - logsumexp(log_weight))
    return weight, float(1 / np.sum(weight * weight))


def mixture_cdf(value, means, sigmas, weights):
    return float(np.dot(weights, ndtr((value - means) / sigmas)))


def mixture_logpdf(value, means, sigmas, weights, observation_sigma=0):
    width = np.hypot(sigmas, observation_sigma)
    z = (value - means) / width
    positive = weights > 0
    return float(logsumexp(np.log(weights[positive]) - np.log(width[positive])
                           - .5 * z[positive] * z[positive]
                           - .5 * np.log(2 * np.pi)))


def mixture_quantiles(means, sigmas, weights, probabilities=(.025, .16, .5, .84, .975)):
    lower = float(np.min(means - 8 * sigmas))
    upper = float(np.max(means + 8 * sigmas))
    return np.array([brentq(lambda x: mixture_cdf(x, means, sigmas, weights) - p,
                            lower, upper, xtol=1e-5)
                     for p in probabilities])


def predict_one(train_colors, train_covariances, train_mg, train_sigma_mg,
                query_color, query_covariance, bandwidth, magnitude_floor,
                observed_mg=None, observed_sigma_mg=0):
    """Predict a full color-conditional luminosity mixture for one star."""
    weight, effective_n = conditional_weights(
        train_colors, train_covariances, query_color, query_covariance,
        bandwidth)
    means = np.asarray(train_mg, float)
    sigmas = np.hypot(np.asarray(train_sigma_mg, float), magnitude_floor)
    quantiles = mixture_quantiles(means, sigmas, weight)
    result = {"quantiles": quantiles, "effective_n": effective_n,
              "max_weight": float(weight.max())}
    if observed_mg is not None:
        result["log_density"] = mixture_logpdf(
            observed_mg, means, sigmas, weight, observed_sigma_mg)
    return result

"""Approximate forward parallax distribution from photometric Mg quantiles."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri


QUANTILE_LEVELS = np.array([.025, .16, .5, .84, .975])


def parallax_forward_check(observed_mas, error_mas, g0, absolute_magnitude_quantiles,
                           grid_size=100):
    """Convolve interpolated model parallax quantiles with Gaia measurement noise.

    End quantiles are held constant beyond 2.5/97.5 percent. This is a
    population diagnostic of the five-quantile model, not a full stellar PDF.
    """
    observed = np.asarray(observed_mas, float)
    error = np.asarray(error_mas, float)
    g0 = np.asarray(g0, float)
    magnitude = np.asarray(absolute_magnitude_quantiles, float)
    if magnitude.shape != (len(observed), 5) or error.shape != observed.shape or g0.shape != observed.shape:
        raise ValueError("Forward-parallax arrays must align")
    if (np.any(~np.isfinite(observed)) or np.any(~np.isfinite(error)) or
            np.any(error <= 0) or np.any(~np.isfinite(g0)) or
            np.any(~np.isfinite(magnitude)) or np.any(np.diff(magnitude, axis=1) < 0)):
        raise ValueError("Nonfinite or unordered parallax-forward inputs")
    levels = (np.arange(grid_size) + .5) / grid_size
    samples = np.array([np.interp(levels, QUANTILE_LEVELS, row) for row in magnitude])
    parallax = 10 ** ((samples - g0[:, None] + 10) / 5)
    mean = np.mean(parallax, axis=1)
    variance = np.var(parallax, axis=1)
    standardized = (observed - mean) / np.sqrt(error**2 + variance)
    pit = np.mean(ndtr((observed[:, None] - parallax) / error[:, None]), axis=1)
    return {"model_parallax_mean_mas": mean,
            "model_parallax_variance_mas2": variance,
            "standardized_residual": standardized,
            "approximate_pit": pit}


def weighted_selected_parallax_check(observed_mas, error_mas, g0,
                                     training_mg, forest_weights,
                                     selection_snr_max=10.0, chunk_size=128):
    """Convolve a discrete forest Mg distribution and condition on Y/sigma < cut.

    The returned selected PIT is conditional only on the explicit observed
    parallax threshold. Other catalog and spectroscopy selection remains part
    of the population being evaluated, rather than the conditional CDF.
    """
    observed = np.asarray(observed_mas, float)
    error = np.asarray(error_mas, float)
    g0 = np.asarray(g0, float)
    mg = np.asarray(training_mg, float)
    weights = np.asarray(forest_weights, float)
    if (observed.ndim != 1 or error.shape != observed.shape or g0.shape != observed.shape
            or mg.ndim != 1 or weights.shape != (len(observed), len(mg))):
        raise ValueError("Forward-parallax distribution arrays must align")
    if (not np.isfinite(observed).all() or not np.isfinite(error).all()
            or not np.isfinite(g0).all() or not np.isfinite(mg).all()
            or not np.isfinite(weights).all() or np.any(error <= 0)
            or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, atol=1e-8)
            or not np.isfinite(selection_snr_max) or selection_snr_max <= 0):
        raise ValueError("Invalid parallax, error or forest weights")
    threshold = selection_snr_max * error
    if np.any(observed >= threshold):
        raise ValueError("Observed parallax fails the stated upper selection")
    result = {name: np.empty(len(observed), float) for name in (
        "model_parallax_mean_mas", "model_parallax_variance_mas2",
        "standardized_residual", "unconditional_pit", "selection_probability",
        "selected_pit", "cdf_at_zero", "cdf_at_five_sigma")}
    for start in range(0, len(observed), chunk_size):
        stop = min(len(observed), start + chunk_size)
        p = 10 ** ((mg[None, :] - g0[start:stop, None] + 10) / 5)
        w = weights[start:stop]
        mean = np.sum(w * p, axis=1)
        variance = np.sum(w * (p - mean[:, None]) ** 2, axis=1)
        cdf_y = np.sum(w * ndtr((observed[start:stop, None] - p) /
                                  error[start:stop, None]), axis=1)
        cdf_t = np.sum(w * ndtr((threshold[start:stop, None] - p) /
                                  error[start:stop, None]), axis=1)
        cdf_0 = np.sum(w * ndtr(-p / error[start:stop, None]), axis=1)
        cdf_5 = np.sum(w * ndtr((5 * error[start:stop, None] - p) /
                                  error[start:stop, None]), axis=1)
        if np.any(cdf_t <= 0):
            raise ValueError("Predicted selection probability underflowed to zero")
        result["model_parallax_mean_mas"][start:stop] = mean
        result["model_parallax_variance_mas2"][start:stop] = variance
        result["standardized_residual"][start:stop] = ((observed[start:stop] - mean) /
            np.sqrt(error[start:stop] ** 2 + variance))
        result["unconditional_pit"][start:stop] = cdf_y
        result["selection_probability"][start:stop] = cdf_t
        result["selected_pit"][start:stop] = np.clip(cdf_y / cdf_t, 0, 1)
        result["cdf_at_zero"][start:stop] = cdf_0
        result["cdf_at_five_sigma"][start:stop] = cdf_5
    return result


def interval_selected_pit(cdf_y, lower_cdf, upper_cdf):
    """PIT conditional on the exact observed-parallax interval."""
    y, lower, upper = (np.asarray(x, float) for x in (cdf_y, lower_cdf, upper_cdf))
    if y.shape != lower.shape or y.shape != upper.shape:
        raise ValueError("CDF arrays must align")
    probability = upper - lower
    if np.any(~np.isfinite(probability)) or np.any(probability <= 0):
        raise ValueError("Selected interval has zero model probability")
    return np.clip((y - lower) / probability, 0, 1), probability


def selected_parallax_null_draws(error_mas, g0, training_mg, forest_weights,
                                 seed, n_sim=1000, selection_snr_max=10):
    """Simulate Gaussian measurements conditional on the observed SNR cut."""
    error, g0, mg, weights = (np.asarray(x, float) for x in
                              (error_mas, g0, training_mg, forest_weights))
    if (error.ndim != 1 or g0.shape != error.shape
            or weights.shape != (len(error), len(mg)) or np.any(error <= 0)
            or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1)):
        raise ValueError("Invalid selected-parallax null inputs")
    rng = np.random.default_rng(seed)
    simulated = np.empty((n_sim, len(error)), float)
    for i in range(len(error)):
        parallax = 10 ** ((mg - g0[i] + 10) / 5)
        upper_probability = ndtr(selection_snr_max - parallax / error[i])
        selected_weights = weights[i] * upper_probability
        total = selected_weights.sum()
        if total <= 0:
            raise ValueError("Selected null probability is zero")
        components = rng.choice(len(mg), n_sim, p=selected_weights / total)
        cdf_draw = rng.random(n_sim) * upper_probability[components]
        simulated[:, i] = parallax[components] + error[i] * ndtri(
            np.clip(cdf_draw, np.finfo(float).tiny, 1 - np.finfo(float).eps))
    return simulated

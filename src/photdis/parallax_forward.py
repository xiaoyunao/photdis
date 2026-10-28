"""Approximate forward parallax distribution from photometric Mg quantiles."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


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

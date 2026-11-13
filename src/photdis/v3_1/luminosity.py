"""Likelihood-level achromatic scale/scatter, distinct from a radial prior."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def luminosity_transform(likelihood, grid, delta, sigma):
    """Convolve last-axis native likelihood, then shift toward larger true mu.

    The same linear operation applies to conditional selection probabilities.
    Zero extension at the declared grid boundary is explicit; callers must
    certify that their likelihood-times-prior support is contained in the grid.
    """
    value = np.asarray(likelihood, float); grid = np.asarray(grid, float)
    if grid.ndim != 1 or len(grid) < 3 or value.shape[-1] != len(grid):
        raise ValueError('Uniform grid aligned with the likelihood required')
    step = grid[1]-grid[0]
    if step <= 0 or not np.allclose(np.diff(grid), step, atol=1e-11, rtol=1e-11):
        raise ValueError('Increasing uniform distance-modulus grid required')
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Finite nonnegative likelihood required')
    if not np.isfinite(delta) or not np.isfinite(sigma) or abs(delta) > 1 or not 0 <= sigma <= 1:
        raise ValueError('Registered delta/sigma bounds exceeded')
    if delta == 0 and sigma == 0: return value.copy()
    blurred = gaussian_filter1d(value, sigma/step, axis=-1, mode='constant', cval=0., truncate=8.) if sigma > 0 else value
    # Interpolate the likelihood at mu-delta. Never interpolate its logarithm:
    # interpolation and prior replacement must preserve this linear operator.
    position = (grid-delta-grid[0])/step
    lo = np.floor(position).astype(int); frac = position-lo; hi = lo+1
    left = np.take(blurred, np.clip(lo, 0, len(grid)-1), axis=-1)
    right = np.take(blurred, np.clip(hi, 0, len(grid)-1), axis=-1)
    return left*(1-frac)*(lo >= 0)*(lo < len(grid))+right*frac*(hi >= 0)*(hi < len(grid))

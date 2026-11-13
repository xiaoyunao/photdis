"""Log-space equivalent of the registered linear luminosity operation."""
import numpy as np
from scipy.special import logsumexp
from numpy.lib.stride_tricks import sliding_window_view


def log_luminosity_transform(loglike, grid, delta, sigma, *, truncate=8.):
    value = np.asarray(loglike,float); grid = np.asarray(grid,float)
    if grid.ndim != 1 or len(grid) < 3 or value.shape[-1] != len(grid): raise ValueError('Aligned grid required')
    step = grid[1]-grid[0]
    if step <= 0 or not np.allclose(np.diff(grid),step,atol=1e-11,rtol=1e-11): raise ValueError('Uniform increasing grid required')
    if np.isnan(value).any() or np.isposinf(value).any(): raise ValueError('Finite or negative-infinite log likelihood required')
    if not np.isfinite(delta) or not np.isfinite(sigma) or abs(delta)>1 or not 0<=sigma<=1 or not 8<=truncate<=16:
        raise ValueError('Registered parameter bounds required')
    if delta == 0 and sigma == 0: return value.copy()
    radius = int(truncate*sigma/step+.5); blurred = value.copy()
    if radius:
        offset = np.arange(-radius,radius+1); kernel = -.5*(offset/(sigma/step))**2; kernel -= logsumexp(kernel)
        padding = [(0,0)]*(value.ndim-1)+[(radius,radius)]
        windows = sliding_window_view(np.pad(value,padding,constant_values=-np.inf),2*radius+1,axis=-1)
        for start in range(0,len(grid),128):
            stop = min(start+128,len(grid))
            blurred[...,start:stop] = logsumexp(windows[...,start:stop,:]+kernel,axis=-1)
    position = (grid-delta-grid[0])/step; lo = np.floor(position).astype(int); hi = lo+1; fraction = position-lo
    with np.errstate(divide='ignore'):
        left = np.take(blurred,np.clip(lo,0,len(grid)-1),axis=-1)+np.log1p(-fraction)
        right = np.take(blurred,np.clip(hi,0,len(grid)-1),axis=-1)+np.log(fraction)
    left[..., (lo<0)|(lo>=len(grid))] = -np.inf; right[..., (hi<0)|(hi>=len(grid))] = -np.inf
    return np.logaddexp(left,right)

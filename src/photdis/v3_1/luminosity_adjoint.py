"""Log-space adjoint of the frozen discrete luminosity shift/convolution."""
import numpy as np
from scipy.special import logsumexp
from numpy.lib.stride_tricks import sliding_window_view


def log_luminosity_adjoint(log_weight, grid, delta, sigma, *, truncate=8.):
    """Move a true-mu integration weight onto the original native grid.

    The forward operation blurs, then interpolates at mu-delta. Its adjoint
    first scatters interpolation weights, then applies the symmetric blur.
    Integration/bin weights belong in log_weight; none are inserted here.
    Leading dimensions are independent weights, e.g. multiple mode basins.
    """
    value=np.asarray(log_weight,float); grid=np.asarray(grid,float)
    if grid.ndim!=1 or len(grid)<3 or value.ndim<1 or value.shape[-1]!=len(grid):
        raise ValueError('Aligned grid required')
    step=grid[1]-grid[0]
    if not np.isfinite(grid).all() or step<=0 or not np.allclose(np.diff(grid),step,atol=1e-11,rtol=1e-11):
        raise ValueError('Uniform increasing grid required')
    if np.isnan(value).any() or np.isposinf(value).any():
        raise ValueError('Finite or negative-infinite log weights required')
    if not np.isfinite(delta) or not np.isfinite(sigma) or abs(delta)>1 or not 0<=sigma<=1 or not 8<=truncate<=16:
        raise ValueError('Registered parameter bounds required')
    if delta==0 and sigma==0:return value.copy()
    position=(grid-delta-grid[0])/step; lo=np.floor(position).astype(int); hi=lo+1; fraction=position-lo
    result=np.full_like(value,-np.inf)
    with np.errstate(divide='ignore'):
        left=value+np.log1p(-fraction); right=value+np.log(fraction)
    # Floating-point positions can share a floor index even on linspace grids.
    # Scatter-add, rather than assignment, preserves those repeated contributions.
    for index,term in [(lo,left),(hi,right)]:
        valid=(index>=0)&(index<len(grid))
        for target,source in zip(result.reshape(-1,len(grid)),term.reshape(-1,len(grid))):
            np.logaddexp.at(target,index[valid],source[valid])
    radius=int(truncate*sigma/step+.5)
    if not radius:return result
    offset=np.arange(-radius,radius+1); kernel=-.5*(offset/(sigma/step))**2; kernel-=logsumexp(kernel)
    padding=[(0,0)]*(value.ndim-1)+[(radius,radius)]
    windows=sliding_window_view(np.pad(result,padding,constant_values=-np.inf),2*radius+1,axis=-1)
    blurred=np.empty_like(result)
    for start in range(0,len(grid),128):
        stop=min(start+128,len(grid)); blurred[...,start:stop]=logsumexp(windows[...,start:stop,:]+kernel,axis=-1)
    return blurred

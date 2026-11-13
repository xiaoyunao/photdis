"""Native class likelihood integrals with exact Gaussian prior withdrawal."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from photdis.v3.likelihood import trapezoid_weights
from .gaussian_entry import transformed_radial_parameters


def class_gaussian_logresponse(loglike,grid,means,sigmas,corrections):
    loglike,grid,corrections=map(lambda x:np.asarray(x,float),(loglike,grid,corrections))
    if loglike.shape!=(3,len(grid)) or corrections.ndim!=3 or corrections.shape[1:]!=(3,2): raise ValueError('Three native class curves and shared corrections required')
    if np.isnan(loglike).any() or np.isposinf(loglike).any(): raise ValueError('Valid native log likelihood required')
    parameters=[transformed_radial_parameters(means,sigmas,p) for p in corrections]
    location=np.array([x[0] for x in parameters]);scale=np.array([x[1] for x in parameters]);logw=np.log(trapezoid_weights(grid))
    result=np.empty(location.shape)
    for c in range(3):
        density=norm.logpdf(grid[None,None],location[:,c,:,None],scale[:,c,:,None])
        result[:,c]=logsumexp(loglike[c][None,None]+density+logw[None,None],axis=-1)
    if not np.isfinite(result).all(): raise ValueError('No finite integrated class response')
    return result

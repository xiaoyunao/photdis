"""True-distance joint integrals after a Gaussian luminosity correction.

The effective native coordinate v=mu-delta-t is not the true distance modulus.
Conditioning the Gaussian prior on v gives the true-distance bin probabilities;
binning v directly would misassign both the shift and scatter uncertainty.
"""
import numpy as np
from scipy.special import log_ndtr,logsumexp
from scipy.stats import norm
from photdis.v3.likelihood import trapezoid_weights
from .gaussian_entry import transformed_radial_parameters


def interval_linear_weights(grid,boundaries):
    """Exact bin weights for the piecewise-linear trapezoidal interpolant."""
    grid=np.asarray(grid,float);boundaries=np.asarray(boundaries,float)
    if grid.ndim!=1 or len(grid)<2 or not np.isfinite(grid).all() or not (np.diff(grid)>0).all():raise ValueError('Increasing finite integration grid required')
    if boundaries.ndim!=1 or not np.isfinite(boundaries).all() or not (np.diff(boundaries)>0).all():raise ValueError('Increasing finite bin boundaries required')
    edges=np.r_[-np.inf,boundaries,np.inf];width=np.diff(grid);result=np.zeros((len(edges)-1,len(grid)))
    for b,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        t0=np.clip((lo-grid[:-1])/width,0,1);t1=np.clip((hi-grid[:-1])/width,0,1)
        square=.5*(t1*t1-t0*t0);result[b,:-1]+=width*(t1-t0-square);result[b,1:]+=width*square
    return result


def normal_log_bin_probability(mean,sigma,boundaries):
    """Stable normal-bin log masses, including intervals deep in either tail."""
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Positive finite conditional width required')
    mean=np.asarray(mean,float);boundaries=np.asarray(boundaries,float)
    edges=np.r_[-np.inf,boundaries,np.inf];z=(edges[:,None]-mean[None])/sigma;result=[]
    for lo,hi in zip(z[:-1],z[1:]):
        upper=np.where(lo>=0,log_ndtr(-lo),log_ndtr(hi))
        lower=np.where(lo>=0,log_ndtr(-hi),log_ndtr(lo))
        with np.errstate(divide='ignore',invalid='ignore'):
            result.append(upper+np.log(-np.expm1(lower-upper)))
    return np.array(result)


def class_gaussian_joint_logresponse(loglike,grid,means,sigmas,corrections,boundaries):
    loglike,grid,means,sigmas,corrections=map(lambda a:np.asarray(a,float),(loglike,grid,means,sigmas,corrections))
    if loglike.shape!=(3,len(grid)) or corrections.ndim!=3 or corrections.shape[1:]!=(3,2):raise ValueError('Aligned native class curves and corrections required')
    if np.isnan(loglike).any() or np.isposinf(loglike).any():raise ValueError('Valid native log likelihood required')
    logw=np.log(trapezoid_weights(grid));result=np.empty((len(corrections),3,len(means),len(boundaries)+1))
    for state,parameter in enumerate(corrections):
        location,scale=transformed_radial_parameters(means,sigmas,parameter)
        for c in range(3):
            delta,scatter=parameter[c]
            for d,(m,s) in enumerate(zip(means,sigmas)):
                density=loglike[c]+norm.logpdf(grid,location[c,d],scale[c,d])
                if scatter==0:
                    weights=interval_linear_weights(grid,np.asarray(boundaries)-delta)
                    with np.errstate(divide='ignore'):value=logsumexp(density[None]+np.log(weights),axis=1)
                else:
                    fraction=s*s/(s*s+scatter*scatter)
                    conditional_mean=m+fraction*(grid-(m-delta))
                    conditional_sigma=s*scatter/np.sqrt(s*s+scatter*scatter)
                    probability=normal_log_bin_probability(conditional_mean,conditional_sigma,boundaries)
                    value=logsumexp(density[None]+logw[None]+probability,axis=1)
                result[state,c,d]=value
    if np.isnan(result).any() or np.isposinf(result).any():raise ValueError('Invalid true-distance response')
    return result

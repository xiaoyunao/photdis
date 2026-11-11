"""Continuous target CDF from exact Gaussian-ratio integrals of stored CDFs.

The reference compression is piecewise constant density. Integration below is
analytic for that representation; native-likelihood compression error remains a
separate measured tolerance, not removed by this operation.
"""
import numpy as np
from scipy.special import ndtr
from .inference import DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA
from .compression import DISTANCE_MU_BOUNDARIES
from .marginal import normal_ratio_maximum


def cdf_product(compressed, model, coefficient, maximum_step=.025):
    if maximum_step <= 0: raise ValueError('Positive integration grid step required')
    means = DISTANCE_BASIS_MEAN; sigmas = DISTANCE_BASIS_SIGMA
    variance = 1/(1/sigmas**2-1/25.)
    centers = variance*(means/sigmas**2-15/25.)
    scales = np.sqrt(variance)
    peaks = np.array([normal_ratio_maximum(m, s) for m, s in zip(means, sigmas)])
    factors = peaks*np.sqrt(2*np.pi*variance)
    lo = float(compressed['mu'].min()); hi = float(compressed['mu'].max())
    boundaries = DISTANCE_MU_BOUNDARIES[(DISTANCE_MU_BOUNDARIES > lo) & (DISTANCE_MU_BOUNDARIES < hi)]
    grid = np.unique(np.r_[compressed['mu'], np.linspace(lo, hi, int(np.ceil((hi-lo)/maximum_step))+1), boundaries])
    z = (grid[None]-centers[:, None])/scales[:, None]
    # Use survival CDF on the positive side to avoid cancellation at far tails.
    integrals = np.where(z[:, :-1] > 0, ndtr(-z[:, :-1])-ndtr(-z[:, 1:]),
                        ndtr(z[:, 1:])-ndtr(z[:, :-1]))*factors[:, None]
    centers_grid = (grid[:-1]+grid[1:])/2
    joint = np.zeros((3, len(grid)-1)); binary = 0.
    component = np.zeros(len(model.weights))
    for j, (k, b, total) in enumerate(zip(compressed['component'], compressed['binary'], compressed['mass'])):
        a, e = compressed['offsets'][j:j+2]
        x = compressed['mu'][a:e].astype(float); y = compressed['cdf'][a:e].astype(float)
        index = np.searchsorted(x, centers_grid, side='right')-1
        valid = (index >= 0) & (index < len(x)-1)
        slope = np.zeros(len(index)); ix = index[valid]
        slope[valid] = total*np.maximum(np.diff(y)[ix], 0)/np.diff(x)[ix]
        value = coefficient[k].T@(integrals*slope[None]/model.weights[k])
        joint += value; component[k] += value.sum()
        if b: binary += value.sum()
    total = joint.sum()
    if not np.isfinite(total) or total <= 0: raise ValueError('No continuous target support')
    joint /= total; cumulative = np.c_[np.zeros(3), np.cumsum(joint, axis=1)]
    cdf = cumulative.sum(axis=0); cdf[-1] = 1.
    quantiles = np.interp([.025, .16, .5, .84, .975], cdf, grid)
    edges = np.r_[grid[0], DISTANCE_MU_BOUNDARIES, grid[-1]]
    bcdf = np.array([np.interp(edges, grid, row, left=0, right=row[-1]) for row in cumulative])
    mass = np.diff(bcdf, axis=1)
    upper = np.max(np.einsum('kdc,d->k', coefficient, peaks)/model.weights)
    omitted = float(compressed['omitted_mass']*upper)
    return dict(grid=grid, class_cdf=cumulative, mu=quantiles, class_distance=mass,
        qnear=float(mass[:, :3].sum()), binary_probability=float(binary/total),
        component_probability=component/total, integral=float(total),
        omitted_posterior_bound=omitted/(total+omitted), joint_prior_ratio_bound=float(upper))

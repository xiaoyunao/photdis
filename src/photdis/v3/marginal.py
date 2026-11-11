"""Observed-star posterior after marginalizing shared population parameters.

Average individually normalized joint posteriors. Averaging priors first is not
valid: each star has a different evidence under each population draw.
"""
import numpy as np
from scipy.stats import norm
from .inference import distance_basis, DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA
from .compression import quadrature, DISTANCE_MU_BOUNDARIES


def normal_ratio_maximum(mean, sigma, base_mean=15., base_sigma=5.):
    if sigma >= base_sigma: raise ValueError('A narrower target Gaussian is required for this finite bound')
    location = (mean/sigma**2-base_mean/base_sigma**2)/(1/sigma**2-1/base_sigma**2)
    return float(np.exp(norm.logpdf(location, mean, sigma)-norm.logpdf(location, base_mean, base_sigma)))


def marginal_coefficients(response, coefficients, draw_weights=None):
    """Return star x kernel x radial x class coefficients and evidence ratios."""
    r = np.asarray(response).sum(axis=3)
    coef = np.asarray(coefficients)
    w = np.ones(len(coef))/len(coef) if draw_weights is None else np.asarray(draw_weights, float)
    if (w < 0).any() or not np.isclose(w.sum(), 1): raise ValueError('Normalized draw weights required')
    evidence = r.reshape(len(r), -1)@coef.sum(axis=3).reshape(len(coef), -1).T
    if (evidence <= 0).any() or not np.isfinite(evidence).all(): raise ValueError('Missing population evidence')
    factors = w[None]/evidence
    average = (factors@coef.reshape(len(coef), -1)).reshape((len(r),)+coef.shape[1:])
    return average, evidence


def marginal_star(compressed, model, coefficient, order=3):
    q = quadrature(compressed, order); k = q['component']; mu = q['mu']
    radial = distance_basis(mu)
    ratio = np.einsum('qdc,dq->qc', coefficient[k], radial)/(
        model.weights[k]*norm.pdf(mu, 15., 5.))[:, None]
    joint = q['mass'][:, None]*ratio
    normalization = joint.sum()
    if normalization <= 0: raise ValueError('Zero marginal posterior support')
    joint /= normalization
    bins = np.searchsorted(DISTANCE_MU_BOUNDARIES, mu, side='right')
    mass = np.array([[joint[bins == b, c].sum() for b in range(6)] for c in range(3)])
    order_mu = np.argsort(mu); cdf = np.cumsum(joint.sum(axis=1)[order_mu]); cdf /= cdf[-1]
    quantiles = np.interp([.025, .16, .5, .84, .975], cdf, mu[order_mu])
    radial_bounds = np.array([normal_ratio_maximum(m, s) for m, s in zip(DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA)])
    # This bounds ALL omitted kernels and all real mu, not only retained nodes.
    upper = np.max(np.einsum('kdc,d->k', coefficient, radial_bounds)/model.weights)
    omitted = float(compressed['omitted_mass']*upper)
    return dict(class_distance=mass, mu=quantiles, qnear=float(mass[:, :3].sum()),
        binary_probability=float(joint[q['binary'] == 1].sum()),
        integral=float(normalization), omitted_posterior_bound=omitted/(normalization+omitted),
        joint_prior_ratio_bound=float(upper))

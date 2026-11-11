"""Conditional photometric-entry response for native grz S/N admission.

Morphology, sky, image-quality masks and reported errors are conditioned-on
covariates. This is not the survey detection/area selection function and cannot
turn observed-catalog counts into an intrinsic Galactic density.
"""
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
from .inference import distance_basis
from .likelihood import trapezoid_weights

OPTICAL = np.array([0, 1, 3])


def effective_errors(observations):
    error = np.asarray(observations['flux_error'])[:, OPTICAL]
    trans = np.asarray(observations['transmission'])[:, OPTICAL]
    value = error/trans
    if not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError('Finite positive native grz error/transmission required')
    return value


def component_entry(model, effective_error, mu, *, draws=128, seed=301501, snr_min=5.):
    """Integrate intrinsic SED uncertainty and native Gaussian measurement noise.

    Conditional on each intrinsic SED draw, independent band noise is integrated
    analytically (CDF), not simulated as a hard threshold. Sobol refinement then
    measures intrinsic-kernel quadrature error. Both binary states are retained.
    """
    if draws < 2 or draws & (draws-1): raise ValueError('Sobol draw count must be a power of two')
    error = np.asarray(effective_error, float)
    if error.shape != (3,) or (error <= 0).any(): raise ValueError('Three positive grz errors required')
    sampler = qmc.Sobol(3, scramble=True, seed=seed)
    u = sampler.random_base2(int(np.log2(draws)))
    z = ndtri(np.clip(u, np.finfo(float).eps, 1-np.finfo(float).eps))
    mean = model.means[:, OPTICAL]
    cov = model.covariances[:, OPTICAL][:, :, OPTICAL]
    chol = np.linalg.cholesky(cov)
    sed = mean[:, None, :]+np.einsum('kij,nj->kni', chol, z)
    amplitude = 10.**(-.4*(np.asarray(mu)-10.))
    result = np.zeros((2, len(mean), len(mu)))
    for b, brightness in enumerate((1., model.binary_brightness_factor)):
        for start in range(0, len(mean), 16):
            stop = min(start+16, len(mean))
            argument = sed[start:stop, :, None, :]*brightness*amplitude[None, None, :, None]/error-snr_min
            result[b, start:stop] = ndtr(argument).prod(axis=-1).mean(axis=1)
    if (result < 0).any() or (result > 1).any(): raise RuntimeError('Invalid entry response')
    return result


def integrate_entry(model, component_response, mu, *, component_tilt=None,
                    binary_fraction=None):
    fraction = model.binary_reference_fraction if binary_fraction is None else binary_fraction
    if not 0 <= fraction <= 1: raise ValueError('Invalid binary fraction')
    curve = (1-fraction)*component_response[0]+fraction*component_response[1]
    prior = model.weights[:, None]*model.classes
    if component_tilt is not None: prior *= np.exp(np.asarray(component_tilt)[:, None])
    prior /= prior.sum(axis=0)[None, :]
    radial = distance_basis(mu)
    response = np.einsum('km,kc,dm,m->cd', curve, prior, radial, trapezoid_weights(mu))
    if (response < 0).any() or (response > 1+1e-10).any(): raise RuntimeError('Invalid integrated entry response')
    return response


def fit_error_contexts(errors, contexts=64, seed=301501):
    """Photometry-only context compression; never use class/distance outcomes."""
    from sklearn.cluster import KMeans
    logerror = np.log(np.asarray(errors))
    if not np.isfinite(logerror).all(): raise ValueError('Invalid error context')
    fit = KMeans(contexts, random_state=seed, n_init=1, max_iter=100).fit(logerror)
    return np.exp(fit.cluster_centers_), fit.labels_


def assign_error_context(errors, centers):
    from scipy.spatial import cKDTree
    distance, index = cKDTree(np.log(centers)).query(np.log(errors))
    return index, distance

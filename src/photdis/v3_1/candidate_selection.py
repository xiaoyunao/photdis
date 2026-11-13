"""Conditional entry including the final candidate's brightness and support cuts.

The original observed selection is evaluated in its original calibration, while
the Gaussian likelihood may use a target-relative calibration draw. Morphology,
sky and image-quality flags remain conditioned covariates, not completeness.
"""
import numpy as np
from scipy.special import log_ndtr, ndtri_exp
from scipy.stats import qmc
from photdis.v3.selection import OPTICAL

MAG_FACTOR = 2.5/np.log(10.)


def candidate_thresholds(raw_error, transmission, calibration_mag, *, r_limit=20., max_mag_error=.1):
    error, trans, cal = (np.asarray(x, float) for x in (raw_error, transmission, calibration_mag))
    if any(x.shape != (3,) for x in (error, trans, cal)) or not all(np.isfinite(x).all() for x in (error, trans, cal)):
        raise ValueError('Three finite native grz errors/transmissions/calibrations required')
    if (error <= 0).any() or (trans <= 0).any() or max_mag_error <= 0 or not np.isfinite(max_mag_error) or not np.isfinite(r_limit):
        raise ValueError('Positive error/transmission and magnitude-error limit required')
    scale = 10.**(.4*cal)/trans
    effective_error = error*scale
    lower = MAG_FACTOR/max_mag_error*effective_error
    lower[1] = max(lower[1], 10.**(.4*(22.5-r_limit))*scale[1])
    return effective_error, lower


def selected_gaussian_probability(mean, covariance, lower, calibration_mag, *, support=None, draws=256, seed=317101):
    """Sequential truncated-Gaussian QMC with the exact observed support test.

    All leading mean/covariance dimensions are batched. A support callable takes
    original extinction-corrected grz magnitudes and returns a Boolean mask.
    Truncation integrates the three flux cuts without rejection of rare tails;
    the remaining hard support boundary must pass independent QMC refinements.
    """
    mean, covariance = np.asarray(mean, float), np.asarray(covariance, float)
    lower, cal = np.asarray(lower, float), np.asarray(calibration_mag, float)
    if mean.shape[-1:] != (3,) or covariance.shape != mean.shape[:-1]+(3, 3):
        raise ValueError('Aligned Gaussian grz means and covariances required')
    if lower.shape != (3,) or cal.shape != (3,) or (lower <= 0).any():
        raise ValueError('Three positive flux thresholds and calibration offsets required')
    if not all(np.isfinite(x).all() for x in (mean, covariance, lower, cal)):
        raise ValueError('Finite conditional Gaussian parameters required')
    if draws < 2 or draws & (draws-1): raise ValueError('Power-of-two Sobol quadrature required')
    leading = mean.shape[:-1]; mean = mean.reshape(-1, 3)
    chol = np.linalg.cholesky(covariance.reshape(-1, 3, 3))
    u = qmc.Sobol(3, scramble=True, seed=seed).random_base2(int(np.log2(draws)))
    logu = np.log(np.clip(u, np.finfo(float).eps, 1-np.finfo(float).eps))
    z = np.zeros((len(mean), draws, 3)); logweight = np.zeros(z.shape[:2])
    for j in range(3):
        conditional_mean = mean[:, j, None]+np.einsum('ni,nsi->ns', chol[:, j, :j], z[:, :, :j])
        logtail = log_ndtr((conditional_mean-lower[j])/chol[:, j, j, None])
        z[:, :, j] = -ndtri_exp(logtail+logu[None, :, j])
        logweight += logtail
    weight = np.exp(logweight)
    if support is not None:
        flux = mean[:, None, :]+np.einsum('nij,nsj->nsi', chol, z)
        if (flux <= 0).any() or not np.isfinite(flux).all():
            raise RuntimeError('Truncated Gaussian did not retain positive selected flux')
        magnitudes = 22.5-2.5*np.log10(flux)+cal
        admitted = np.asarray(support(magnitudes.reshape(-1, 3)))
        if admitted.dtype.kind != 'b' or admitted.shape != (len(mean)*draws,):
            raise ValueError('Support callback must return one Boolean per grz row')
        weight *= admitted.reshape(weight.shape)
    return weight.mean(axis=1).reshape(leading)


def candidate_component_entry(model, raw_error, transmission, calibration_mag, mu, *, support, draws=256, seed=317101):
    """Binary x SED kernel x distance native candidate-entry probabilities."""
    error, lower = candidate_thresholds(raw_error, transmission, calibration_mag)
    means = model.means[:, OPTICAL]; covariance = model.covariances[:, OPTICAL][:, :, OPTICAL]
    amplitude = 10.**(-.4*(np.asarray(mu)-10.))
    result = np.empty((2, len(means), len(amplitude)))
    for b, factor in enumerate((1., model.binary_brightness_factor)):
        a = amplitude*factor
        # Bound the exact nearest-neighbor callback's memory footprint.
        for k in range(len(means)):
            mean = means[k][None]*a[:, None]
            cov = covariance[k][None]*a[:, None, None]**2+np.diag(error**2)[None]
            result[b, k] = selected_gaussian_probability(mean, cov, lower, calibration_mag,
                support=support, draws=draws, seed=seed)
    return result

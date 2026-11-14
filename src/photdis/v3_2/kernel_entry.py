"""Kernel-resolved entry integrals under the existing v3.1 admission model."""
import numpy as np
from scipy.special import log_ndtr, ndtri, ndtri_exp
from scipy.stats import qmc
from photdis.v3.selection import OPTICAL
from photdis.v3_1.candidate_selection import candidate_thresholds
from photdis.v3_1.gaussian_importance import gaussian_ratio_coefficients


def kernel_entry(model, error, transmission, calibration, means, sigmas, *,
                 support=None, draws=524288, seed=319711):
    """Integrate entry for each kernel and radial basis before applying any tilt.

    The kernel proposal combines the average class-conditional mixture and a
    uniform floor. Radial draws use N(15, 6). Normalize radial importance
    weights within each sampled kernel, producing a self-normalized estimator.
    Its finite-sample ratio bias and integration error are checked by refinement.
    Return the maximum discarded integrand as a conservative pruning bound.
    The model must include the local dust covariance.
    """
    if draws < 2 or draws & (draws - 1):
        raise ValueError('Power-of-two Sobol size required')
    error, lower = candidate_thresholds(error, transmission, calibration)
    if support is None:
        lower = 5 * error
    conditional = model.weights[:, None] * model.classes
    conditional /= conditional.sum(axis=0)
    kernel_count = len(model.weights)
    proposal = .9 * conditional.mean(axis=1) + .1 / kernel_count
    cumulative = np.cumsum(proposal)
    cumulative[-1] = 1.
    uniform = qmc.Sobol(5, scramble=True, seed=seed).random_base2(draws.bit_length() - 1)
    uniform = np.clip(uniform, np.finfo(float).eps, 1 - np.finfo(float).eps)
    index = np.searchsorted(cumulative, uniform[:, 0], side='right')
    mu = 15 + 6 * ndtri(uniform[:, 1])
    log_uniform = np.log(uniform[:, 2:])
    amplitude = 10 ** (-.4 * (mu - 10))
    intrinsic = model.means[index][:, OPTICAL]
    optical_covariance = model.covariances[:, OPTICAL][:, :, OPTICAL]
    covariance = optical_covariance[index]
    value = np.zeros(draws)
    binary = model.binary_reference_fraction
    for fraction, factor in ((1 - binary, 1.), (binary, model.binary_brightness_factor)):
        scaled = amplitude * factor
        mean = intrinsic * scaled[:, None]
        chol = np.linalg.cholesky(covariance * scaled[:, None, None] ** 2 + np.diag(error ** 2))
        z = np.zeros((draws, 3))
        log_weight = np.zeros(draws)
        for band in range(3):
            center = mean[:, band] + np.einsum('ni,ni->n', chol[:, band, :band], z[:, :band])
            tail = log_ndtr((center - lower[band]) / chol[:, band, band])
            log_weight += tail
            if band < 2 or support is not None:
                z[:, band] = -ndtri_exp(tail + log_uniform[:, band])
        weight = np.exp(log_weight)
        if support is not None:
            active = weight > 1e-14
            admitted = np.zeros(draws, dtype=bool)
            if active.any():
                flux = mean[active] + np.einsum('nij,nj->ni', chol[active], z[active])
                if (flux <= 0).any() or not np.isfinite(flux).all():
                    raise ValueError('Invalid selected flux')
                keep = np.asarray(support(22.5 - 2.5 * np.log10(flux) + calibration))
                if keep.dtype.kind != 'b' or keep.shape != (int(active.sum()),):
                    raise ValueError('Boolean candidate support required')
                admitted[active] = keep
            weight *= admitted
        value += fraction * weight

    a, b, c, _ = gaussian_ratio_coefficients(means, sigmas)
    entry = np.empty((kernel_count, len(means)))
    for j in range(len(means)):
        ratio = np.exp((a[j] * mu + b[j]) * mu + c[j])
        denominator = np.bincount(index, weights=ratio, minlength=kernel_count)
        numerator = np.bincount(index, weights=ratio * value, minlength=kernel_count)
        if (denominator <= 0).any():
            raise ValueError('Missing radial proposal support for a kernel')
        entry[:, j] = numerator / denominator
    bound = np.full_like(entry, 1e-14 if support is not None else 0.)
    if not np.isfinite(entry).all() or (entry < 0).any() or (entry > 1 + 1e-12).any():
        raise ValueError('Invalid kernel selection integral')
    return entry, bound, np.bincount(index, minlength=kernel_count)

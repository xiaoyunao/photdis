"""Low-dimensional color–luminosity model with a forward parallax likelihood."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr

LOG_PAR = np.log(10) / 5


def design_matrix(gr, rz):
    x = np.asarray(gr, float) - 1.0
    y = np.asarray(rz, float) - 0.5
    return np.column_stack((np.ones(len(x)), x, y, x * x))


def expected_parallax(params, data, with_feh):
    basis = design_matrix(data["gr0"], data["rz0"])
    mg = basis @ params[:4]
    if with_feh:
        mg += params[4] * (data["feh_cal"] + 1.0)
    return 10 ** ((mg - data["g0"] + 10) / 5)


def effective_mag_variance(params, data, with_feh, feh_floor):
    """Propagate shared g/r/z errors through f(g-r,r-z)-g and [Fe/H]."""
    x = data["gr0"] - 1.0
    dgr = params[1] + 2 * params[3] * x
    drz = params[2]
    photo = ((dgr - 1) * data["sigma_g"]) ** 2
    photo += ((drz - dgr) * data["sigma_r"]) ** 2
    photo += (drz * data["sigma_z"]) ** 2
    beta = params[4] if with_feh else 0.0
    iron = beta**2 * (data["desi_feh_err"]**2 + feh_floor**2)
    scatter = params[-1]**2
    return photo + iron + scatter


def log_probability(params, data, with_feh, feh_floor=0.1, nodes=24):
    """Convolve positive latent parallax with Gaia noise using local quadrature.

    Recenter the integration proposal at each star's approximate posterior
    mode. Fixed nodes around the prior miss a narrow high-S/N likelihood peak.
    """
    p = expected_parallax(params, data, with_feh)
    sigma_mag = np.sqrt(effective_mag_variance(params, data, with_feh, feh_floor))
    a = LOG_PAR * sigma_mag
    y = data["p_corr"]
    sigma = data["gaia_parallax_error"]
    lower = np.full(len(p), -30.0)
    upper = np.full(len(p), 30.0)
    for _ in range(48):
        middle = (lower + upper) / 2
        q = p * np.exp(a * middle)
        grad = -middle + (y - q) * q * a / sigma**2
        lower = np.where(grad > 0, middle, lower)
        upper = np.where(grad > 0, upper, middle)
    mode = (lower + upper) / 2
    q = p * np.exp(a * mode)
    curvature = 1 - (y - 2*q) * q * a**2 / sigma**2
    proposal_sigma = np.clip(1 / np.sqrt(np.maximum(curvature, .1)), .02, 2)
    abscissa, weight = hermgauss(nodes)
    latent = mode[:, None] + np.sqrt(2) * proposal_sigma[:, None] * abscissa
    true_parallax = p[:, None] * np.exp(a[:, None] * latent)
    logpdf = (-0.5 * ((y[:, None] - true_parallax) / sigma[:, None]) ** 2
              - np.log(sigma[:, None]) - 0.5 * np.log(2 * np.pi))
    logprior_minus_proposal = (-0.5 * latent**2 + np.log(proposal_sigma[:, None])
                               + 0.5 * ((latent - mode[:, None]) /
                                        proposal_sigma[:, None])**2)
    return (logsumexp(logpdf + logprior_minus_proposal
                      + np.log(weight)[None, :], axis=1) - 0.5 * np.log(np.pi))


def predictive_cdf(params, data, with_feh, feh_floor=0.1, nodes=129):
    p = expected_parallax(params, data, with_feh)
    sigma_mag = np.sqrt(effective_mag_variance(params, data, with_feh, feh_floor))
    abscissa, weight = hermgauss(nodes)
    true_parallax = p[:, None] * np.exp(LOG_PAR * np.sqrt(2) * sigma_mag[:, None]
                                       * abscissa[None, :])
    z = ((data["p_corr"][:, None] - true_parallax) /
         data["gaia_parallax_error"][:, None])
    return (ndtr(z) * weight).sum(axis=1) / np.sqrt(np.pi)


def fit(data, with_feh, feh_floor=0.1, beta_prior_sigma=2.0):
    high = data["p_corr"] / data["gaia_parallax_error"] >= 20
    if high.sum() < 20:
        raise ValueError("Insufficient high-S/N stars to initialize the forward model")
    mg = data["g0"][high] - 10 + 5 * np.log10(data["p_corr"][high])
    basis = design_matrix(data["gr0"][high], data["rz0"][high])
    if with_feh:
        basis = np.column_stack((basis, data["feh_cal"][high] + 1))
    initial = np.r_[np.linalg.lstsq(basis, mg, rcond=None)[0], 0.3]
    nparam = 6 if with_feh else 5
    bounds = [(-5, 15), (-10, 15), (-10, 15), (-10, 10)]
    if with_feh:
        bounds.append((-3, 3))
    bounds.append((0.03, 2))

    def objective(params):
        likelihood = log_probability(params, data, with_feh, feh_floor)
        penalty = 0.5 * (params[4] / beta_prior_sigma)**2 if with_feh else 0
        return -likelihood.sum() + penalty

    result = minimize(objective, initial[:nparam], bounds=bounds,
                      method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-8})
    if not result.success:
        raise RuntimeError(f"Forward-likelihood optimization failed: {result.message}")
    return result.x, {"objective": float(result.fun), "iterations": int(result.nit),
                      "high_snr_initialization_count": int(high.sum())}

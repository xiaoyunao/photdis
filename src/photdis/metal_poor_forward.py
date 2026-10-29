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


def log_probability(params, data, with_feh, feh_floor=0.1, nodes=12):
    """Integrate the positive lognormal parallax over latent magnitude scatter."""
    p = expected_parallax(params, data, with_feh)
    sigma_mag = np.sqrt(effective_mag_variance(params, data, with_feh, feh_floor))
    abscissa, weight = hermgauss(nodes)
    true_parallax = p[:, None] * np.exp(LOG_PAR * np.sqrt(2) * sigma_mag[:, None]
                                       * abscissa[None, :])
    sigma = data["gaia_parallax_error"][:, None]
    y = data["p_corr"][:, None]
    logpdf = (-0.5 * ((y - true_parallax) / sigma) ** 2
              - np.log(sigma) - 0.5 * np.log(2 * np.pi))
    return logsumexp(logpdf + np.log(weight)[None, :], axis=1) - 0.5 * np.log(np.pi)


def predictive_cdf(params, data, with_feh, feh_floor=0.1, nodes=12):
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

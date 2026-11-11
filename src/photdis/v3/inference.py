"""Continuous joint photometric posterior for the v3 empirical SED model."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from .likelihood import flux_loglikelihood, adaptive_posterior, trapezoid_weights, normalize_joint

# Smooth overlapping continuous-distance bases, distinct from reporting bins.
DISTANCE_BASIS_MEAN = np.array([8., 11.5, 14., 16.5, 20.5])
DISTANCE_BASIS_SIGMA = np.array([3., 1.5, 1.4, 1.8, 3.5])


def distance_basis(mu):
    return norm.pdf(np.asarray(mu)[None, :], DISTANCE_BASIS_MEAN[:, None], DISTANCE_BASIS_SIGMA[:, None])


def kernel_loglike(model, observation, mu, *, bands=None, luminosity_shift=None,
                   calibration_mag=None, candidates=None):
    """Single/binary components have a shared underlying SED, including covariance."""
    mask = observation['band_mask'].copy()
    if bands is not None: mask &= np.asarray(bands, bool)
    indices = np.arange(len(model.weights)) if candidates is None else np.asarray(candidates, int)
    mean = model.means[indices, :6].copy(); cov = model.covariances[indices, :6, :6].copy()
    if luminosity_shift is not None:
        factor = 10.**(.4*np.asarray(luminosity_shift)[indices])
        mean *= factor[:, None]; cov *= factor[:, None, None]**2
    values = []
    for brightness in (1., model.binary_brightness_factor):
        values.append(flux_loglikelihood(observation['flux'], observation['flux_error'], mask,
            observation['transmission'], mean*brightness, cov*brightness**2, mu,
            calibration_mag=calibration_mag, system=str(observation['photometric_system']),
            template_system=model.system, upper_limit=observation.get('upper_limit')))
    return np.stack(values), indices


def reference_log_density(model, observation, mu, *, bands=None, luminosity_shift=None,
                          calibration_mag=None, candidates=None, prior_mean=15., prior_sigma=5.):
    likelihood, ix = kernel_loglike(model, observation, mu, bands=bands,
        luminosity_shift=luminosity_shift, calibration_mag=calibration_mag, candidates=candidates)
    binary = np.array([1-model.binary_reference_fraction, model.binary_reference_fraction])
    logp = likelihood+np.log(binary)[:, None, None]+np.log(model.weights[ix])[None, :, None]
    logp += norm.logpdf(mu, prior_mean, prior_sigma)[None, None, :]
    return logp.reshape(-1, len(mu))


def infer_reference(model, observation, *, bands=None, candidates=None, tolerance=.001,
                    initial_step=.15, luminosity_shift=None, calibration_mag=None):
    p = adaptive_posterior(lambda mu: reference_log_density(model, observation, mu, bands=bands,
        luminosity_shift=luminosity_shift, calibration_mag=calibration_mag, candidates=candidates),
        step=initial_step, probability_tolerance=tolerance)
    ix = np.arange(len(model.weights)) if candidates is None else np.asarray(candidates, int)
    joint = p.distance_bins().reshape(2, len(ix), -1)
    class_distance = np.einsum('bkd,kc->cd', joint, model.classes[ix])
    component_mass = joint.sum(axis=(0, 2))
    modes, mode_mass = p.modes()
    negative_support = float(component_mass@model.negative_intrinsic_flux_probability[ix].max(axis=1))
    return dict(posterior=p, mu_quantiles=p.quantiles(), qnear=float(p.cdf(15.).sum()),
                class_distance=class_distance, binary_probability=float(joint[1].sum()),
                component_probability=component_mass, modes=modes, mode_mass=mode_mass,
                negative_intrinsic_support=negative_support, converged=p.convergence['converged'])


def basis_response(model, observation, mu, *, bands=None, candidates=None,
                   component_tilt=None, luminosity_shift=None, calibration_mag=None):
    """Unselected native-data response for class x continuous radial basis.

    Component tilt changes within-class empirical SED weights, separately from
    class/distance weights. Reference normalization and actual priors are returned.
    This never uses the target's spectroscopic class or Gaia parallax.
    """
    likelihood, ix = kernel_loglike(model, observation, mu, bands=bands,
        candidates=candidates, luminosity_shift=luminosity_shift, calibration_mag=calibration_mag)
    binary = np.array([1-model.binary_reference_fraction, model.binary_reference_fraction])
    lk = logsumexp(likelihood+np.log(binary)[:, None, None], axis=0)
    prior = model.weights[:, None]*model.classes
    if component_tilt is not None:
        prior *= np.exp(np.asarray(component_tilt)[:, None])
    prior /= prior.sum(axis=0)[None, :]
    w = trapezoid_weights(mu); radial = distance_basis(mu)
    logs = np.empty((3, len(radial)))
    for c in range(3):
        for d in range(len(radial)):
            logs[c, d] = logsumexp(lk+np.log(prior[ix, c])[:, None]+np.log(np.maximum(radial[d]*w, 1e-300))[None, :])
    return logs, prior


def target_posterior(model, observation, mu, theta, *, bands=None, component_tilt=None):
    theta = np.asarray(theta).reshape(3, len(DISTANCE_BASIS_MEAN))
    if (theta < 0).any() or not np.isclose(theta.sum(), 1): raise ValueError('Normalized joint class/distance-basis prior required')
    likelihood, ix = kernel_loglike(model, observation, mu, bands=bands)
    binary = np.array([1-model.binary_reference_fraction, model.binary_reference_fraction])
    prior = model.weights[:, None]*model.classes
    if component_tilt is not None: prior *= np.exp(np.asarray(component_tilt)[:, None])
    prior /= prior.sum(axis=0)[None, :]
    radial = theta@distance_basis(mu)
    # Explicit b x component x class x continuous-mu posterior support.
    with np.errstate(divide='ignore'):
        lp = (likelihood[:, :, None, :]+np.log(binary)[:, None, None, None]
              +np.log(prior)[None, :, :, None]+np.log(radial)[None, None, :, :])
    posterior = normalize_joint(mu, lp.reshape(-1, len(mu)))
    return posterior, dict(component_given_class=prior, class_distance_weights=theta,
                           radial_means=DISTANCE_BASIS_MEAN, radial_sigmas=DISTANCE_BASIS_SIGMA,
                           binary_weights=binary)

"""Truth-assisted mechanism controls, never a photometry-only production route."""
import copy
import numpy as np
from scipy.special import logsumexp
from scipy.signal import find_peaks
from scipy.stats import norm
from photdis.v3.inference import kernel_loglike, distance_basis
from photdis.v3.likelihood import adaptive_posterior


def condition_feh(model, value, error):
    """p(flux | measured FeH, kernel) and log p(measured FeH | kernel).

    FeH is coordinate6 in the joint empirical SED. Conditioning changes both
    flux moments and kernel evidence, including flux/FeH cross covariance.
    The measured FeH error is integrated, not treated as exact latent truth.
    The input model is immutable; no class or source selection is done here.
    """
    if not np.isfinite(value) or not np.isfinite(error) or error < 0:
        raise ValueError('Finite FeH and nonnegative measurement error required')
    m = copy.deepcopy(model)
    variance = model.covariances[:, 6, 6] + error**2
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError('Positive FeH marginal variance required')
    cross = model.covariances[:, :6, 6]
    m.means[:, :6] += cross*(value-model.means[:, 6])[:, None]/variance[:, None]
    m.covariances[:, :6, :6] -= cross[:, :, None]*cross[:, None, :]/variance[:, None, None]
    log_evidence = norm.logpdf(value, model.means[:, 6], np.sqrt(variance))
    return m, log_evidence


def control_log_density(model, observation, mu, *, reference_class=None,
                        reference_feh=None, reference_feh_error=None,
                        bands=None, calibration_mag=None, theta=None,
                        component_tilt=None, error_scale=None, binary_fraction=None):
    """Explicit binary x kernel x class x mu density for paired controls.

    theta=None reproduces the frozen reference prior exactly. A target theta
    must belong to a stated shared draw; averaging theta before inference is
    not marginalization over shared draws. Optional noise changes are diagnostic.
    """
    if reference_class is not None and reference_class not in (0, 1, 2):
        raise ValueError('Reliable reference class0/1/2 required; unknown is not a class')
    if (reference_feh is None) != (reference_feh_error is None):
        raise ValueError('FeH value and measurement error must be supplied together')
    conditioned, feh_likelihood = (model, np.zeros(len(model.weights)))
    if reference_feh is not None:
        conditioned, feh_likelihood = condition_feh(model, reference_feh, reference_feh_error)
    obs = dict(observation)
    if error_scale is not None:
        scale = np.asarray(error_scale, float)
        if scale.shape != (6,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Six positive finite error multipliers required')
        obs['flux_error'] = np.asarray(observation['flux_error'])*scale
    likelihood, _ = kernel_loglike(conditioned, obs, mu, bands=bands, calibration_mag=calibration_mag)
    fraction = model.binary_reference_fraction if binary_fraction is None else binary_fraction
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Binary fraction must be a probability')
    binary = np.array([1-fraction, fraction])
    prior = model.weights[:, None]*model.classes
    if theta is None:
        if component_tilt is not None:
            raise ValueError('Reference control cannot silently acquire a target component tilt')
        radial = np.broadcast_to(norm.pdf(mu, 15., 5.), (3, len(mu)))
    else:
        theta = np.asarray(theta, float)
        if theta.shape != (3, 5) or not np.isfinite(theta).all() or (theta < 0).any() or not np.isclose(theta.sum(), 1):
            raise ValueError('Normalized class x five-basis target weights required')
        if component_tilt is not None:
            tilt = np.asarray(component_tilt, float)
            if tilt.shape != model.weights.shape or not np.isfinite(tilt).all():
                raise ValueError('Finite per-component log tilt required')
            prior *= np.exp(tilt-tilt.max())[:, None]
        if (prior.sum(axis=0) <= 0).any():
            raise ValueError('Target class has no empirical support')
        prior /= prior.sum(axis=0)[None, :]
        radial = theta@distance_basis(mu)
    if reference_class is not None:
        prior = prior.copy()
        prior[:, np.arange(3) != reference_class] = 0
    with np.errstate(divide='ignore'):
        lp = (likelihood[:, :, None, :] + np.log(binary)[:, None, None, None]
              + np.log(prior)[None, :, :, None] + feh_likelihood[None, :, None, None]
              + np.log(radial)[None, None, :, :])
    return lp.reshape(-1, len(mu))


def mode_attribution(posterior, components, parent_component=None, minimum_mass=.05):
    """Integrate each marginal mode's basin over the full latent joint density.

    Binary/class fractions are responsibilities, not a unique physical cause.
    Weak basins are reported separately; a small mode is not proved numerical.
    Parent components combine broad discrepancy copies with their original SED.
    """
    if posterior.joint_density.shape[0] != 2*components*3:
        raise ValueError('Expected binary x kernel x class joint density')
    parent = np.arange(components) if parent_component is None else np.asarray(parent_component)
    if parent.shape != (components,) or not np.issubdtype(parent.dtype, np.integer) or (parent < 0).any():
        raise ValueError('Nonnegative integer parent component IDs required')
    density = posterior.joint_density.sum(axis=0)
    peaks, _ = find_peaks(density)
    if not len(peaks): peaks = np.array([density.argmax()])
    valleys = [a+np.argmin(density[a:b+1]) for a, b in zip(peaks[:-1], peaks[1:])]
    bounds = np.r_[posterior.mu[0], posterior.mu[np.asarray(valleys, int)], posterior.mu[-1]]
    result = []
    for peak, lower, upper in zip(peaks, bounds[:-1], bounds[1:]):
        joint = (posterior.cdf(upper)-posterior.cdf(lower)).reshape(2, components, 3)
        mass = float(joint.sum())
        if mass <= 0: continue
        component = joint.sum(axis=(0, 2))/mass
        parent_mass = np.bincount(parent, weights=component)
        result.append(dict(mu=float(posterior.mu[peak]), lower=float(lower), upper=float(upper),
            mass=mass, class_probability=(joint.sum(axis=(0, 1))/mass).tolist(),
            binary_probability=float(joint[1].sum()/mass),
            dominant_parent=int(parent_mass.argmax()), dominant_parent_probability=float(parent_mass.max()),
            effective_parent_components=float(1/(parent_mass@parent_mass)),
            weak_basin=mass <= minimum_mass))
    return result


def infer_control(model, observation, *, initial_step=.1, tolerance=.001, **control):
    fn = lambda mu: control_log_density(model, observation, mu, **control)
    p = adaptive_posterior(fn, step=initial_step, probability_tolerance=tolerance)
    joint = p.distance_bins().reshape(2, len(model.weights), 3, -1)
    conditioned = model
    if control.get('reference_feh') is not None:
        conditioned, _ = condition_feh(model, control['reference_feh'], control['reference_feh_error'])
    # This is a component-marginal prior-support proxy, not posterior latent
    # P(any true flux<0 | observations). Name it explicitly to prevent misuse.
    negative = norm.cdf(-conditioned.means[:, :6]/np.sqrt(
        np.diagonal(conditioned.covariances[:, :6, :6], axis1=1, axis2=2)))
    component_mass = joint.sum(axis=(0, 2, 3))
    return dict(posterior=p, mu=p.quantiles(), qnear=float(p.cdf(15.).sum()),
        class_distance=joint.sum(axis=(0, 1)), binary_probability=float(joint[1].sum()),
        component_probability=component_mass,
        kernel_negative_support_proxy_by_band=component_mass@negative,
        modes=mode_attribution(p, len(model.weights), getattr(model, 'parent_component', None)),
        full_logpdf=logsumexp(fn(p.mu), axis=0)-p.log_evidence,
        diagnostic_only=any(control.get(k) is not None for k in (
            'reference_class', 'reference_feh', 'error_scale', 'binary_fraction')))

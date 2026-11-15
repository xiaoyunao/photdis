"""Class-conditional likelihood curves and their joint distance posterior."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.v3.inference import kernel_loglike


def class_log_likelihood(model, observation, mu):
    """Return log p(flux | class, mu), marginalizing kernels and binary state."""
    kernel_class_mass = model.weights[:, None] * model.classes
    class_mass = kernel_class_mass.sum(axis=0)
    if (kernel_class_mass < 0).any() or (class_mass <= 0).any() or not np.isclose(class_mass.sum(), 1.):
        raise ValueError('A normalized mixture with support in every class is required')
    logs, _ = kernel_loglike(model, observation, mu)
    f = model.binary_reference_fraction
    with np.errstate(divide='ignore'):
        log_binary = np.log([1-f, f])
        log_kernel_given_class = np.log(kernel_class_mass / class_mass)
    by_kernel = logsumexp(logs + log_binary[:, None, None], axis=0)
    return logsumexp(by_kernel[:, None, :] + log_kernel_given_class[:, :, None], axis=0)


def reference_log_curves(log_likelihood, mu, class_prior):
    """Apply the common nominal reference prior to class-specific curves."""
    log_likelihood = np.asarray(log_likelihood, float)
    class_prior = np.asarray(class_prior, float)
    if (class_prior.shape != (3,) or log_likelihood.shape != (3, len(mu))
            or not np.isfinite(class_prior).all() or (class_prior <= 0).any()
            or not np.isclose(class_prior.sum(), 1.)):
        raise ValueError('Three class likelihood curves and a normalized prior are required')
    return log_likelihood + np.log(class_prior)[:, None] + norm.logpdf(mu, 15., 5.)

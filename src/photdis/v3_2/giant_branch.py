"""One fixed posterior class-allocation adjustment for Photdis v3.2."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from photdis.v3.likelihood import trapezoid_weights


def reference_class_curves(model, grid, native):
    """Class-conditioned evidence curves under the frozen common distance prior."""
    frequency = model.weights @ model.classes
    within_class = model.weights[:, None] * model.classes / frequency
    with np.errstate(divide='ignore'):
        curves = logsumexp(native[:, None, :] + np.log(within_class)[:, :, None], axis=0)
    return curves + norm.logpdf(grid, 15., 5.)


def adjust_giant_branch(grid, class_logpdf, reference_curves, frequency):
    """Average giant log odds and preserve each normalized class curve."""
    grid = np.asarray(grid, dtype=float)
    target = np.asarray(class_logpdf, dtype=float)
    reference = np.asarray(reference_curves, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    if (grid.ndim != 1 or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any()
            or target.shape != (3, len(grid)) or reference.shape != target.shape
            or frequency.shape != (3,) or not np.isfinite(frequency).all()
            or (frequency <= 0).any()):
        raise ValueError('Aligned class curves, grid and positive class frequencies required')
    log_width = np.log(trapezoid_weights(grid))
    target_mass = logsumexp(target + log_width, axis=1)
    reference_mass = logsumexp(reference + log_width, axis=1) + np.log(frequency)
    if not (np.isfinite(target_mass).all() and np.isfinite(reference_mass).all()):
        raise ValueError('Positive distance support in every class is required')
    nongiant = [0, 2]
    target_nongiant = logsumexp(target_mass[nongiant])
    reference_nongiant = logsumexp(reference_mass[nongiant])
    log_odds = .5 * ((target_mass[1] - target_nongiant)
                     + (reference_mass[1] - reference_nongiant))
    mass = np.empty(3)
    mass[1] = -np.logaddexp(0., -log_odds)
    mass[nongiant] = target_mass[nongiant] - target_nongiant - np.logaddexp(0., log_odds)
    return target + (mass - target_mass)[:, None]

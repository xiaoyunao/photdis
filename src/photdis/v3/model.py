"""Versioned inference assembly of fitted SEDs and constrained discrepancy mode."""
import copy
import numpy as np
from scipy.special import ndtr

INFERENCE_VERSION = 'v3.0-native-continuous-discrepancy01'


def inference_model(fitted, discrepancy_fraction=.01):
    """Add a fixed broad SED discrepancy channel, not a fourth physical class.

    Known binary brightening is a separate latent variable. Discrepancy broadens
    color/luminosity kernels and carries an explicit posterior warning. It is not
    interpreted as a nonstellar classification or permission to trust bad fits.
    The fraction/width are fixed development choices, not selected on final data.
    """
    if hasattr(fitted, 'inference_version'): raise ValueError('Inference model already assembled')
    if not 0 < discrepancy_fraction < .1: raise ValueError('Constrained discrepancy fraction required')
    model = copy.deepcopy(fitted); k = len(fitted.weights)
    broad = fitted.covariances*4
    for j in range(6): broad[:, j, j] += (.5*abs(fitted.means[:, j]))**2
    model.means = np.concatenate([fitted.means, fitted.means])
    model.covariances = np.concatenate([fitted.covariances, broad])
    model.weights = np.r_[(1-discrepancy_fraction)*fitted.weights, discrepancy_fraction*fitted.weights]
    model.classes = np.concatenate([fitted.classes, fitted.classes])
    model.parent_component = np.tile(np.arange(k, dtype='i2'), 2)
    model.discrepancy_channel = np.r_[np.zeros(k, bool), np.ones(k, bool)]
    model.negative_intrinsic_flux_probability = ndtr(-model.means[:, :6]/np.sqrt(np.diagonal(model.covariances[:, :6, :6], axis1=1, axis2=2)))
    model.components = 2*k; model.base_components = k
    model.inference_version = INFERENCE_VERSION
    model.discrepancy_fraction = discrepancy_fraction
    for key in ('effective_reference_sources', 'reference_source_fraction', 'missing_class_reference_fraction'):
        if hasattr(fitted, key): setattr(model, key, np.concatenate([getattr(fitted, key)]*2))
    return model

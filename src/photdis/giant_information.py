"""Bounded native-system giant selection and hidden-abundance baselines."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BANDS = ('g0', 'r0', 'z0', 'w10', 'w20')
ERRORS = ('sigma_g', 'sigma_r', 'sigma_z', 'sigma_w1', 'sigma_w2')
COLOR_MATRIX = np.array([[1., -1., 0., 0., 0.], [0., 1., -1., 0., 0.],
                         [0., 0., 1., -1., 0.], [0., 0., 0., 1., -1.]])
DIMENSIONS = {'A': 2, 'B': 3, 'C': 4}


def colors_and_covariance(magnitudes, errors, dimension):
    # Slice bands before multiplication: unused NaNs must not poison grz.
    transform = COLOR_MATRIX[:dimension, :dimension + 1]
    mag = np.asarray(magnitudes, float)[..., :dimension + 1]
    sigma = np.asarray(errors, float)[..., :dimension + 1]
    color = mag @ transform.T
    covariance = np.einsum('ab,...b,cb->...ac', transform, sigma**2, transform)
    return color, covariance


def band_realizations(magnitudes, errors, draws=16, seed=20261031):
    if draws < 2 or draws % 2:
        raise ValueError('Use an even positive number of antithetic draws')
    mag, sigma = np.asarray(magnitudes, float), np.asarray(errors, float)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(len(mag), draws // 2, 5))
    noise = np.concatenate([noise, -noise], axis=1)
    return mag[:, None, :] + sigma[:, None, :] * noise


def linked_groups(ls_id, source_id, desi_id):
    """Connect every shared identity; transitive links never cross CV folds."""
    n = len(ls_id)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for values, kind in ((ls_id, 'ls'), (source_id, 'gaia'), (desi_id, 'desi')):
        seen = {}
        for i, value in enumerate(values):
            if kind == 'desi':
                value = str(value.decode() if isinstance(value, bytes) else value).strip()
                if value in ('', '0', '-1', 'None'):
                    continue
            elif value <= 0:
                continue
            if value in seen:
                a, b = root(i), root(seen[value])
                parent[max(a, b)] = min(a, b)
            else:
                seen[value] = i
    return np.array([root(i) for i in range(n)])


def wilson(success, total, z=1.959963984540054):
    if total == 0:
        return {'n': 0, 'success': 0, 'rate': None, 'interval95': [None, None]}
    p = success / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return {'n': int(total), 'success': int(success), 'rate': float(p),
            'interval95': [float(center-half), float(center+half)]}


def class_metrics(label, score, threshold=.5):
    label, score = np.asarray(label), np.asarray(score)
    selected = score >= threshold
    giant, dwarf = label == 1, label == 0
    return {'giants': wilson(int((selected & giant).sum()), int(giant.sum())),
            'dwarfs': wilson(int((selected & dwarf).sum()), int(dwarf.sum())),
            'selected': int(selected.sum()), 'n': len(label)}


def purity(tpr, fpr, fractions):
    if tpr is None or fpr is None:
        return [None] * len(fractions)
    pi = np.asarray(fractions)
    numerator = pi * tpr
    den = numerator + (1-pi) * fpr
    return [float(a/b) if b > 0 else None for a, b in zip(numerator, den)]


def fit_classifier(magnitudes, errors, labels, dimension, *, C=1., draws=16, seed=0):
    labels = np.asarray(labels, int)
    if set(labels) != {0, 1}:
        raise ValueError('Both conservative classes are required')
    nominal, _ = colors_and_covariance(magnitudes, errors, dimension)
    poly = PolynomialFeatures(2, include_bias=False)
    scale = StandardScaler().fit(poly.fit_transform(nominal))
    realized = band_realizations(magnitudes, errors, draws, seed)
    colors, _ = colors_and_covariance(realized, np.zeros_like(realized), dimension)
    features = scale.transform(poly.transform(colors.reshape(-1, dimension)))
    counts = np.bincount(labels)
    weights = len(labels) / (2 * counts[labels] * draws)
    model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', tol=1e-7)
    model.fit(features, np.repeat(labels, draws), sample_weight=np.repeat(weights, draws))
    if model.n_iter_[0] >= 1000:
        raise RuntimeError('Fixed classifier did not converge')
    return {'poly': poly, 'scale': scale, 'model': model, 'dimension': dimension,
            'lower': nominal.min(axis=0), 'upper': nominal.max(axis=0)}


def predict_classifier(model, magnitudes, errors, *, draws=16, seed=0):
    dim = model['dimension']
    realized = band_realizations(magnitudes, errors, draws, seed)
    colors, _ = colors_and_covariance(realized, np.zeros_like(realized), dim)
    x = model['scale'].transform(model['poly'].transform(colors.reshape(-1, dim)))
    score = model['model'].predict_proba(x)[:, 1].reshape(len(magnitudes), draws)
    nominal, _ = colors_and_covariance(magnitudes, errors, dim)
    outside = np.any((nominal < model['lower']) | (nominal > model['upper']), axis=1)
    return score.mean(axis=1), score.std(axis=1), outside


def abundance_error(feh_error, teff, teff_error, floor=.1):
    """Giant-branch author correction; derivative with respect to Teff."""
    x = np.log10(np.asarray(teff)/5000)/.1
    derivative = (-.809 + 2*.441*x) / (.1*np.log(10)*np.asarray(teff))
    random = np.hypot(feh_error, derivative*np.asarray(teff_error))
    return np.hypot(random, floor), random


def fit_abundance(colors, covariance, feh, feh_error, *, ridge=1., floor=.05):
    """Gaussian latent conditional density; observed-label errors added once.

    The linear conditional mean permits exact first-order color-error variance.
    This is a limited forward error-propagation model, not a deconvolved color
    population model. No held-out abundance is accepted by this function.
    """
    x, cov = np.asarray(colors, float), np.asarray(covariance, float)
    y, sy = np.asarray(feh, float), np.asarray(feh_error, float)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .05)
    design = np.column_stack([np.ones(len(x)), (x-center)/scale])
    v = cov / scale[None, :, None] / scale[None, None, :]

    def objective(theta):
        beta, tau = theta[:-1], np.exp(theta[-1])
        vb = v @ beta[1:]
        variance = sy**2 + tau**2 + np.sum(vb*beta[1:], axis=1)
        residual = y - design @ beta
        nll = .5*np.sum(np.log(2*np.pi*variance) + residual**2/variance)
        nll += .5*ridge*np.sum(beta[1:]**2)
        response = 1/variance - residual**2/variance**2
        grad = -design.T @ (residual/variance)
        grad[1:] += np.sum(vb*response[:, None], axis=0) + ridge*beta[1:]
        return nll/len(y), np.r_[grad, tau**2*np.sum(response)]/len(y)

    initial = np.r_[y.mean(), np.zeros(x.shape[1]), np.log(max(y.std(), floor))]
    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      bounds=[(None, None)]*(len(initial)-1)+[(np.log(floor), np.log(5.))],
                      options={'maxiter': 2000, 'maxls': 50, 'ftol': 1e-11, 'gtol': 1e-7})
    if not result.success:
        raise RuntimeError('Abundance fit failed: '+str(result.message))
    return {'center': center, 'scale': scale, 'beta': result.x[:-1],
            'sigma': float(np.exp(result.x[-1])), 'objective': float(result.fun),
            'gradient_max': float(np.max(np.abs(result.jac))), 'n': len(y)}


def predict_abundance(model, colors, covariance):
    x, cov = np.asarray(colors), np.asarray(covariance)
    b = model['beta'][1:] / model['scale']
    mean = model['beta'][0] + (x-model['center']) @ b
    sigma = np.sqrt(model['sigma']**2 + np.einsum('i,nij,j->n', b, cov, b))
    return mean, sigma


def abundance_metrics(observed, error, mean, sigma):
    width = np.hypot(sigma, error)
    standardized = (observed-mean)/width
    logpdf = -.5*standardized**2 - np.log(width) - .5*np.log(2*np.pi)
    return {'n': len(observed), 'mean_log_density': float(logpdf.mean()),
            'coverage68': wilson(int((np.abs(standardized)<=.9944578832).sum()), len(observed)),
            'coverage95': wilson(int((np.abs(standardized)<=1.959963985).sum()), len(observed)),
            'latent_width68': float(np.median(2*.9944578832*sigma)),
            'observed_width68': float(np.median(2*.9944578832*width)),
            'pit_mean': float(ndtr(standardized).mean())}, logpdf


def field_bootstrap(values, draws=1000, seed=0):
    values = np.asarray(values, float)
    if not len(values):
        return {'n_fields': 0, 'mean': None, 'interval95': [None, None]}
    sample = np.random.default_rng(seed).choice(values, (draws, len(values)), replace=True).mean(axis=1)
    return {'n_fields': len(values), 'mean': float(values.mean()),
            'interval95': np.quantile(sample, [.025, .975]).tolist(),
            'positive_fraction': float(np.mean(values>0))}

"""Controlled negative-population experiment; historical models stay unchanged."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from .giant_information import colors_and_covariance


def training_weights(labels, reference_n, draws=16):
    labels = np.asarray(labels, int)
    if set(labels) != {0, 1} or reference_n <= 0:
        raise ValueError('Require both classes and positive reference weight')
    return reference_n / (2 * np.bincount(labels)[labels] * draws)


def preprocessing(bands, errors, dimension):
    colors, _ = colors_and_covariance(bands, errors, dimension)
    poly = PolynomialFeatures(2, include_bias=False)
    scale = StandardScaler().fit(poly.fit_transform(colors))
    return dict(poly=poly, scale=scale, dimension=dimension,
                lower=colors.min(axis=0), upper=colors.max(axis=0))


def fit_controlled(prep, realized, labels, reference_n):
    draws = realized.shape[1]
    dim = prep['dimension']
    colors, _ = colors_and_covariance(realized, np.zeros_like(realized), dim)
    features = prep['scale'].transform(prep['poly'].transform(colors.reshape(-1, dim)))
    weights = training_weights(labels, reference_n, draws)
    model = LogisticRegression(C=1., max_iter=1000, solver='lbfgs', tol=1e-7)
    model.fit(features, np.repeat(labels, draws), sample_weight=np.repeat(weights, draws))
    if model.n_iter_[0] >= 1000:
        raise RuntimeError('Fixed classifier did not converge')
    return dict(prep, model=model, reference_n=int(reference_n),
                total_weight=float(weights.sum()*draws),
                class_weight_totals=[float(weights[np.asarray(labels)==v].sum()*draws) for v in (0, 1)])


def field_roles(fields, position):
    fields = list(sorted(fields))
    if len(fields) < 5:
        raise ValueError('At least five fields needed')
    test = fields[position]
    calibration = [fields[(position+1)%len(fields)], fields[(position+2)%len(fields)]]
    train = [f for f in fields if f not in [test, *calibration]]
    return train, calibration, test

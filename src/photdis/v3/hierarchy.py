"""Shared sky shrinkage without counting the same catalog twice.

The global mixture is a hypermean, not a second fit likelihood. Each object's
likelihood occurs only in its owning field. Standard-normal latent coordinates
make the regularization and conditional field variation explicit.
"""
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from .target import population_objective, sed_tilt_features


class FieldPopulation:
    def __init__(self, response, selection, model, field, case_weights=None,
                 global_sigma=.7, field_sigma=.35, beta_sigma=.3):
        self.model = model
        self.features, _ = sed_tilt_features(model)
        self.c = model.classes.shape[1]; self.d = response.shape[2]
        self.k = self.c*self.d-1
        self.fields, owner = np.unique(field, return_inverse=True)
        self.parts = []
        weights = np.ones(len(field)) if case_weights is None else np.asarray(case_weights)
        if not np.isfinite(weights).all() or (weights <= 0).any(): raise ValueError('Positive case weights required')
        # Keep effective evidence at the actual sample size, not the expanded parent.
        weights = weights/weights.mean()
        for j in range(len(self.fields)):
            use = owner == j
            self.parts.append((response[use], selection[use], weights[use]))
        frequency = model.weights@model.classes
        base = (frequency[:, None]*np.full((1, self.d), 1/self.d)).ravel()
        self.center = np.log(base[:-1]/base[-1])
        self.global_sigma = global_sigma; self.field_sigma = field_sigma; self.beta_sigma = beta_sigma
        self.dimension = self.k+len(self.features[0])+len(self.fields)*self.k
        self.case_weight_sum = float(weights.sum())
        self.effective_case_count = float(weights.sum()**2/(weights@weights))

    def decode(self, z):
        z = np.asarray(z)
        g = self.center+self.global_sigma*z[:self.k]
        b = self.beta_sigma*z[self.k:self.k+2]
        local = g[None]+self.field_sigma*z[self.k+2:].reshape(len(self.fields), self.k)
        theta = softmax(np.c_[local, np.zeros(len(local))], axis=1).reshape(len(local), self.c, self.d)
        return theta, b, local, g

    def objective(self, z):
        _, beta, local, _ = self.decode(z)
        value = .5*np.dot(z, z); gradient = np.array(z, copy=True)
        local_gradient = np.empty_like(local); beta_gradient = np.zeros(2)
        for j, (response, selection, weight) in enumerate(self.parts):
            val, grad = population_objective(np.r_[local[j], beta], response, selection,
                self.model.weights, self.model.classes, self.features, self.center,
                np.inf, np.inf, weight)
            value += val; local_gradient[j] = grad[:self.k]; beta_gradient += grad[self.k:]
        gradient[:self.k] += self.global_sigma*local_gradient.sum(axis=0)
        gradient[self.k:self.k+2] += self.beta_sigma*beta_gradient
        gradient[self.k+2:] += self.field_sigma*local_gradient.ravel()
        return float(value), gradient

    def fit(self):
        fit = minimize(self.objective, np.zeros(self.dimension), jac=True, method='BFGS',
                       options=dict(maxiter=500, gtol=1e-5))
        if not fit.success and abs(fit.jac).max() > 1e-3: raise RuntimeError('Hierarchical MAP: '+fit.message)
        # BFGS covariance is only an HMC metric; no reported interval uses it alone.
        covariance = (fit.hess_inv+fit.hess_inv.T)/2
        eigen, vectors = np.linalg.eigh(covariance)
        covariance = (vectors*np.clip(eigen, 1e-5, 10.))@vectors.T
        return dict(map=fit.x, metric=covariance, objective=float(fit.fun),
                    gradient_max=float(abs(fit.jac).max()))

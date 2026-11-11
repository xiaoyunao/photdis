"""Condition shared population draws on additional observed photometry.

The population posterior learned from a target subsample cannot be reused as
unchanged weights for a new catalog and called a full joint posterior. Every
new object's selection-normalized evidence updates the SAME global weights.
Previously fitted identities must contribute zero additional likelihood. The
result remains a finite quadrature approximation and requires an ESS audit.
"""
import numpy as np
from scipy.special import logsumexp


class CatalogConditioning:
    def __init__(self, draw_ids, prior_weights):
        self.draw_ids = np.asarray(draw_ids, dtype='i8')
        w = np.asarray(prior_weights, float)
        if w.shape != self.draw_ids.shape or len(w) < 2 or len(np.unique(self.draw_ids)) != len(w):
            raise ValueError('Unique common global draw IDs required')
        if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError('Finite nonnegative quadrature weights required')
        with np.errstate(divide='ignore'):
            self.log_prior = np.log(w/w.sum())
        self.log_update = np.zeros(len(w))
        self.new_objects = 0
        self.reused_objects = 0

    def add(self, log_evidence, selection_probability, already_fitted, draw_ids):
        """Inputs are draw x object; evidence includes response-specific constants."""
        if not np.array_equal(draw_ids, self.draw_ids):
            raise ValueError('Chunks must preserve the shared draw ordering')
        logl = np.asarray(log_evidence, float)
        alpha = np.asarray(selection_probability, float)
        old = np.asarray(already_fitted, bool)
        if logl.ndim != 2 or logl.shape != alpha.shape or logl.shape[0] != len(self.draw_ids) or old.shape != (logl.shape[1],):
            raise ValueError('Expected draw x object likelihood and selection')
        if not np.isfinite(logl).all() or not np.isfinite(alpha).all() or (alpha <= 0).any() or (alpha > 1+1e-10).any():
            raise ValueError('Finite likelihood and positive entry probability required')
        self.log_update += (logl[:, ~old]-np.log(alpha[:, ~old])).sum(axis=1)
        self.new_objects += int((~old).sum())
        self.reused_objects += int(old.sum())

    def result(self):
        logw = self.log_prior+self.log_update
        normalization = logsumexp(logw)
        weights = np.exp(logw-normalization)
        return dict(draw_ids=self.draw_ids.copy(), weights=weights,
            log_likelihood_update=self.log_update.copy(), log_predictive_evidence=float(normalization),
            effective_sample_size=float(1/(weights@weights)), maximum_weight=float(weights.max()),
            new_objects=self.new_objects, reused_objects=self.reused_objects)

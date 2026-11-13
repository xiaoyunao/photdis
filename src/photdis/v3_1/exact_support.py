"""Exact Euclidean tenth-neighbor QRF support with a compact query backend."""
import numpy as np
from scipy.spatial import cKDTree


class ExactColorSupport:
    def __init__(self, model):
        metric = getattr(model.nn, 'effective_metric_', None)
        if metric != 'euclidean': raise ValueError('Only the original Euclidean support metric is allowed')
        self.model = model
        self.tree = cKDTree(np.asarray(model.nn._fit_X, float))
        if len(self.tree.data) < 10: raise ValueError('At least ten original support points required')

    def __call__(self, magnitudes):
        model = self.model; x = model.feature_matrix(magnitudes)
        inside = ((x >= model.lo) & (x <= model.hi)).all(axis=1)
        answer = np.zeros(len(x),bool)
        if inside.any():
            z = (x[inside]-model.mean)/model.scale
            # k=[10] returns the tenth distance only; eps=0 forbids approximation.
            distance = self.tree.query(z,k=[10],eps=0.,workers=1)[0][:,0]
            answer[inside] = distance <= model.radius
        return answer

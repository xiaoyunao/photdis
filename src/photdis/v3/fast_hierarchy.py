"""Algebraically identical hierarchy with cached matrix-vector contractions."""
import numpy as np
from scipy.special import softmax
from .hierarchy import FieldPopulation


class FastFieldPopulation(FieldPopulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matrices = [(np.ascontiguousarray(r.reshape(len(r), -1), dtype=float),
                          np.ascontiguousarray(a.reshape(len(a), -1), dtype=float), w)
                         for r, a, w in self.parts]

    def objective(self, z):
        theta, beta, _, _ = self.decode(z)
        prior = softmax(np.log(self.model.weights)[:, None]+np.log(self.model.classes)
                        +(self.features@beta)[:, None], axis=0)
        derivative = prior[:, :, None]*(self.features[:, None, :]-(prior.T@self.features)[None])
        value = .5*np.dot(z, z); gradient = np.array(z, copy=True)
        local_gradient = np.zeros((len(self.fields), self.k)); beta_gradient = np.zeros(2)
        for j, (r, a, w) in enumerate(self.matrices):
            coef = prior@theta[j]
            total = r@coef.ravel(); entry = a@coef.ravel()
            if (total <= 0).any() or (entry <= 0).any(): raise ValueError('Zero population support')
            value -= w@(np.log(total)-np.log(entry))
            g = ((w/entry)@a-(w/total)@r).reshape(coef.shape)
            grad_theta = theta[j]*(prior.T@g)
            # The softmax common term is zero analytically; subtract its numeric
            # residual to preserve the full chain rule in finite precision.
            grad_theta -= theta[j]*grad_theta.sum()
            local_gradient[j] = grad_theta.ravel()[:-1]
            beta_gradient += np.einsum('kd,kcf,cd->f', g, derivative, theta[j], optimize=True)
        gradient[:self.k] += self.global_sigma*local_gradient.sum(axis=0)
        gradient[self.k:self.k+2] += self.beta_sigma*beta_gradient
        gradient[self.k+2:] += self.field_sigma*local_gradient.ravel()
        return float(value), gradient

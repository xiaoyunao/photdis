"""Conditional-entry interpolation with exact common-error scale translation.

Only two grz error ratios need profile interpolation. Multiplying all errors by
s is exactly a +2.5 log10(s) translation in distance modulus inside the entry
curve. This avoids approximating a third error dimension by nearest centers.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, cKDTree
from scipy.stats import norm
from .selection import gaussian_component_entry
from .inference import DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA
from .likelihood import trapezoid_weights


def error_profiles(errors):
    errors = np.asarray(errors)
    if not np.isfinite(errors).all() or (errors <= 0).any(): raise ValueError('Positive error covariates required')
    return np.log(errors[:, [0, 2]]/errors[:, 1, None]), 2.5*np.log10(errors[:, 1])


def profile_centers(errors, centers=64):
    from sklearn.cluster import KMeans
    profile, _ = error_profiles(errors)
    unique = np.unique(profile, axis=0)
    if len(unique) <= centers: return unique
    middle = KMeans(centers, n_init=1, max_iter=100, random_state=302201).fit(profile).cluster_centers_
    hull = ConvexHull(unique)
    # Include observed hull vertices so all construction inputs have interpolation
    # support; test/new-sky profiles outside this hull remain explicitly flagged.
    return np.unique(np.r_[middle, unique[hull.vertices]], axis=0)


def profile_table(model, profile, *, draws=128, step=.1, shift_step=.05):
    mu = np.arange(-15., 45.+step/2, step)
    shift = np.arange(-10., 10.+shift_step/2, shift_step)
    errors = np.array([np.exp(profile[0]), 1., np.exp(profile[1])])
    response = gaussian_component_entry(model, errors, mu, draws=draws)
    curve = (1-model.binary_reference_fraction)*response[0]+model.binary_reference_fraction*response[1]
    radial = norm.pdf(mu[None, None, :],
        DISTANCE_BASIS_MEAN[:, None, None]+shift[None, :, None],
        DISTANCE_BASIS_SIGMA[:, None, None])
    value = (curve*trapezoid_weights(mu))@radial.reshape(-1, len(mu)).T
    alpha = value.reshape(len(model.weights), len(DISTANCE_BASIS_MEAN), len(shift))
    if (alpha < 0).any() or (alpha > 1+1e-10).any(): raise RuntimeError('Invalid entry probability')
    return shift, alpha


@dataclass
class SelectionTable:
    profiles: np.ndarray
    shift: np.ndarray
    alpha: np.ndarray

    def evaluate(self, errors):
        profiles, shifts = error_profiles(errors)
        if (shifts < self.shift[0]).any() or (shifts > self.shift[-1]).any():
            raise ValueError('Error-scale translation outside certified grid')
        nearest_distance, nearest = cKDTree(self.profiles).query(profiles)
        indices = nearest[:, None]; weights = np.ones((len(profiles), 1))
        inside = nearest_distance < 1e-10
        if len(self.profiles) >= 3 and np.linalg.matrix_rank(self.profiles-self.profiles.mean(axis=0)) == 2:
            tri = Delaunay(self.profiles); simplex = tri.find_simplex(profiles, tol=1e-10)
            inside = simplex >= 0
            indices = np.tile(nearest[:, None], (1, 3)); weights = np.zeros((len(profiles), 3)); weights[:, 0] = 1
            j = np.flatnonzero(inside); simplex_j = simplex[j]
            barycentric = np.einsum('nij,nj->ni', tri.transform[simplex_j, :2], profiles[j]-tri.transform[simplex_j, 2])
            barycentric = np.c_[barycentric, 1-barycentric.sum(axis=1)]
            barycentric = np.maximum(barycentric, 0); barycentric /= barycentric.sum(axis=1)[:, None]
            indices[j] = tri.simplices[simplex_j]; weights[j] = barycentric
        lower = np.clip(np.searchsorted(self.shift, shifts, side='right')-1, 0, len(self.shift)-2)
        fraction = (shifts-self.shift[lower])/(self.shift[lower+1]-self.shift[lower])
        out = np.zeros((len(profiles), self.alpha.shape[1], self.alpha.shape[2]))
        for j in range(indices.shape[1]):
            lo = self.alpha[indices[:, j], :, :, lower]
            hi = self.alpha[indices[:, j], :, :, lower+1]
            out += weights[:, j, None, None]*((1-fraction[:, None, None])*lo+fraction[:, None, None]*hi)
        return out, dict(inside_profile_hull=inside, nearest_profile_distance=nearest_distance,
                         scale_shift=shifts)

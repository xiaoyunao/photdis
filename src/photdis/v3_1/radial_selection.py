"""Class-conditional entry curves for arbitrary radial priors; no second division."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from photdis.v3.selection_table import error_profiles


@dataclass
class RadialSelectionTable:
    profiles: np.ndarray
    mu: np.ndarray
    entry: np.ndarray

    def evaluate(self, errors, grid):
        profiles,shifts = error_profiles(errors); grid = np.asarray(grid,float)
        if self.entry.shape != (len(self.profiles),3,len(self.mu)):
            raise ValueError('Expected profile x class x mu entry table')
        if not np.isfinite(self.entry).all() or (self.entry < 0).any() or (self.entry > 1+1e-12).any():
            raise ValueError('Finite conditional entry probability table required')
        query = grid[None,:]+shifts[:,None]
        if query.min() < self.mu[0] or query.max() > self.mu[-1]:
            raise ValueError('Shifted radial grid outside entry table support')
        distance,nearest = cKDTree(self.profiles).query(profiles)
        vertices = nearest[:,None]; weights = np.ones((len(profiles),1)); inside = distance < 1e-10
        if len(self.profiles) >= 3 and np.linalg.matrix_rank(self.profiles-self.profiles.mean(axis=0)) == 2:
            triangulation = Delaunay(self.profiles); simplex = triangulation.find_simplex(profiles,tol=1e-10)
            inside = simplex >= 0; vertices = np.tile(nearest[:,None],(1,3)); weights = np.zeros((len(profiles),3)); weights[:,0] = 1
            j = np.flatnonzero(inside); t = triangulation.transform[simplex[j]]
            bary = np.einsum('nij,nj->ni',t[:,:2],profiles[j]-t[:,2]); bary = np.c_[bary,1-bary.sum(axis=1)]
            bary = np.maximum(bary,0); bary /= bary.sum(axis=1)[:,None]
            vertices[j] = triangulation.simplices[simplex[j]]; weights[j] = bary
        result = np.zeros((len(profiles),3,len(grid)))
        for i in range(len(profiles)):
            for vertex,weight in zip(vertices[i],weights[i]):
                if not weight: continue
                for c in range(3): result[i,c] += weight*np.interp(query[i],self.mu,self.entry[vertex,c])
        return result,dict(inside_profile_hull=inside,nearest_profile_distance=distance,scale_shift=shifts)

from types import SimpleNamespace
import numpy as np
from sklearn.neighbors import NearestNeighbors
from photdis.baseline_v1 import support
from photdis.v3_1.exact_support import ExactColorSupport


def test_exact_support_matches_original_color_boundary_and_duplicates():
    rng = np.random.default_rng(319201)
    points = np.r_[rng.normal(size=(150,2)),np.zeros((12,2))]
    model = SimpleNamespace(nn=NearestNeighbors(n_neighbors=11,n_jobs=1).fit(points),
        mean=np.array([.3,-.2]),scale=np.array([.7,1.2]),lo=np.array([-2.,-2.]),hi=np.array([2.,2.]),radius=.7,
        feature_matrix=lambda m:np.diff(-np.asarray(m),axis=1))
    colors = np.r_[rng.uniform(-3,3,size=(4000,2)),model.lo[None],model.hi[None],model.mean[None]]
    mags = np.c_[np.zeros(len(colors)),-np.cumsum(colors,axis=1)]+18.
    np.testing.assert_array_equal(ExactColorSupport(model)(mags),support(model,mags))

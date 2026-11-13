from types import SimpleNamespace
import numpy as np
from photdis.v3_1.exact_support import ExactColorSupport
from photdis.v3_1.certified_support import CertifiedColorSupport


def test_lipschitz_support_certificates_preserve_boundary_and_interior_decisions():
    rng=np.random.default_rng(319601)
    points=np.r_[rng.normal(size=(180,2))*.2,rng.normal(size=(120,2))*.3+2]
    model=SimpleNamespace(dim=2,mean=np.array([.2,-.1]),scale=np.array([.7,1.3]),lo=np.array([-1.,-1.]),hi=np.array([3.,4.]),
        radius=.19,nn=SimpleNamespace(effective_metric_='euclidean',_fit_X=points),feature_matrix=lambda x:x)
    fast=CertifiedColorSupport(model,resolution=64); exact=ExactColorSupport(model)
    x=rng.uniform(model.lo-.1,model.hi+.1,size=(20000,2))
    x=np.r_[x,model.lo[None],model.hi[None],points*model.scale+model.mean]
    np.testing.assert_array_equal(fast(x),exact(x))
    assert fast.certified>0 and fast.fallback>0


def test_radius_ties_and_duplicate_training_points_use_exact_fallback():
    model=SimpleNamespace(dim=2,mean=np.zeros(2),scale=np.ones(2),lo=np.array([-2.,-2.]),hi=np.array([2.,2.]),
        radius=1.,nn=SimpleNamespace(effective_metric_='euclidean',_fit_X=np.zeros((10,2))),feature_matrix=lambda x:x)
    x=np.array([[1.,0.],[np.nextafter(1.,0.),0.],[np.nextafter(1.,2.),0.],[0.,0.],[1.5,1.5]])
    fast=CertifiedColorSupport(model,resolution=32)
    np.testing.assert_array_equal(fast(x),ExactColorSupport(model)(x))
    assert fast.fallback>=3

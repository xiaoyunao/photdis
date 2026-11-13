from types import SimpleNamespace
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
from photdis.v3_1.structure_entry import proposal_entry
from photdis.v3_1.gaussian_importance import shared_gaussian_entry
from photdis.v3_1.candidate_selection import candidate_thresholds


def model():
    return SimpleNamespace(means=np.full((1,7),20.),covariances=np.eye(7)[None]*.04,
        weights=np.ones(1),classes=np.full((1,3),1/3),binary_reference_fraction=.17,binary_brightness_factor=2.)


def test_general_proposal_matches_original_broad_estimator_at_identical_sobol_draws():
    args=(model(),[.2,.3,.25],[.9,.95,.97],[.01,-.01,.02],[11.,15.],[1.,3.])
    expected,_=shared_gaussian_entry(*args,np.zeros((1,3,2)),draws=8192,seed=320211)
    actual,_=proposal_entry(*args,proposal_mean=15,proposal_sigma=6,draws=8192)
    np.testing.assert_allclose(actual,expected[0],atol=2e-14)


def test_narrow_proposal_cancels_against_independent_normal_integral():
    raw=np.array([.2,.3,.25]);trans=np.array([.9,.95,.97]);cal=np.array([.01,-.01,.02])
    error,_=candidate_thresholds(raw,trans,cal);lower=5*error;means=np.array([12.5,13.,13.5]);sigmas=np.full(3,.02)
    expected=[]
    for m,s in zip(means,sigmas):
        answer=0.
        for fraction,factor in [(.83,1.),(.17,2.)]:
            def f(z):
                a=10**(-.4*(m+s*z-10))*factor
                return norm.pdf(z)*ndtr((20*a-lower)/np.sqrt(.04*a*a+error**2)).prod()
            answer+=fraction*quad(f,-10,10,epsabs=1e-12)[0]
        expected.append(answer)
    actual,audit=proposal_entry(model(),raw,trans,cal,means,sigmas,proposal_mean=13,proposal_sigma=.8,draws=131072)
    np.testing.assert_allclose(actual,np.broadcast_to(expected,actual.shape),atol=3e-6,rtol=0)
    assert np.isfinite(audit['maximum_radial_density_ratio']).all()

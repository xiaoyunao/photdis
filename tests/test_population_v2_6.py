import numpy as np
from photdis.population_v2_6 import responsibilities,latent_counts,response_logposterior


def test_observed_responsibility_matches_bayes_enumeration():
    r=np.array([[.45,.05],[.1,.4]])
    p=responsibilities(np.array([[.3,.7]]),r,[0,1])
    np.testing.assert_allclose(p[0],[[.135/.205,.07/.205],[.015/.295,.28/.295]])
    assert abs(p[0,:,0].mean()-.3)>.05


def test_shared_parameter_preserves_count_variance():
    eta=np.tile([[.05,.95],[.95,.05]],(8192,1))
    p,n,_=latent_counts(eta,np.ones((2,1)),np.zeros(20,int))
    np.testing.assert_allclose(p,.5,atol=1e-14)
    assert np.all(n.sum(axis=1)==20)
    # Exact total variance: N p(1-p)+N(N-1) Var(theta).
    expected=20*.25+20*19*.45**2
    assert abs(n[:,0].var()-expected)<2
    assert n[:,0].var()>10*20*.25


def test_response_weights_include_target_evidence():
    eta=np.ones((8,1))
    rs=np.array([[[.9,.1]],[[.1,.9]]])
    w,rw,z=response_logposterior([2,0],eta,rs)
    np.testing.assert_allclose(rw,[[81/82,1/82]])
    np.testing.assert_allclose(w,1/8)
    assert abs(rw[0,0]-.5)>.4


def test_zero_probability_observation_fails():
    import pytest
    with pytest.raises(ValueError):
        responsibilities(np.ones((1,2))*.5,np.zeros((2,1)),[0])

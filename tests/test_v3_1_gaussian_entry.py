from types import SimpleNamespace
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
from photdis.v3_1.gaussian_entry import transformed_radial_parameters,class_gaussian_entry
from photdis.v3_1.candidate_selection import candidate_thresholds


def test_gaussian_prior_withdrawal_equals_analytic_luminosity_convolution():
    means=np.array([11.,15.]);sigmas=np.array([1.,2.])
    correction=np.array([[.4,.3],[-.2,.8],[0.,0.]])
    m,s=transformed_radial_parameters(means,sigmas,correction)
    # Native likelihood is an unnormalized Gaussian centered at12 withwidth.7.
    expected=norm.pdf(means[None],12+correction[:,0,None],np.sqrt(sigmas[None]**2+.7**2+correction[:,1,None]**2))
    actual=norm.pdf(m,12,np.sqrt(s*s+.7**2))
    np.testing.assert_allclose(actual,expected,atol=1e-15)


def test_joint_entry_matches_independent_one_dimensional_quadrature():
    model=SimpleNamespace(means=np.full((1,7),20.),covariances=np.eye(7)[None]*.04,
        weights=np.ones(1),classes=np.full((1,3),1/3),binary_reference_fraction=.17,binary_brightness_factor=2.)
    raw=np.array([.2,.3,.25]);trans=np.array([.9,.95,.97]);cal=np.array([.01,-.01,.02])
    error,lower=candidate_thresholds(raw,trans,cal);means=np.array([11.]);sigmas=np.array([1.])
    expected=0.
    for fraction,factor in [(0.83,1.),(.17,2.)]:
        def integrand(mu):
            a=10**(-.4*(mu-10))*factor
            return norm.pdf(mu,11,1)*ndtr((20*a-lower)/np.sqrt(.04*a*a+error**2)).prod()
        expected+=fraction*quad(integrand,0,25,epsabs=1e-11,points=[10,11,12,13])[0]
    value,audit=class_gaussian_entry(model,raw,trans,cal,means,sigmas,candidate=True,
        support=lambda x:np.ones(len(x),bool),draws=32768,seed=319701)
    np.testing.assert_allclose(value,expected,atol=2e-5,rtol=0)
    assert audit['pruning_bound'].max()<1e-14

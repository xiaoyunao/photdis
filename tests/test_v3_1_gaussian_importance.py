from types import SimpleNamespace
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import ndtr
from photdis.v3_1.gaussian_importance import shared_gaussian_entry,gaussian_ratio_coefficients
from photdis.v3_1.gaussian_entry import transformed_radial_parameters
from photdis.v3_1.candidate_selection import candidate_thresholds


def test_radial_density_ratio_has_a_global_bound():
    m=np.array([7.,14.,21.5]);s=np.array([3.2,1.4,3.7]);a,b,c,bound=gaussian_ratio_coefficients(m,s)
    vertex=-b/(2*a)
    np.testing.assert_allclose(np.exp(a*vertex**2+b*vertex+c),bound,rtol=1e-14)
    grid=np.linspace(-40,70,10001)
    ratio=np.exp(norm.logpdf(grid[None],m[:,None],s[:,None])-norm.logpdf(grid,15,6))
    assert (ratio<=bound[:,None]*(1+1e-12)).all()


def test_shared_proposal_cancels_for_multiple_class_calibration_draws():
    model=SimpleNamespace(means=np.full((1,7),20.),covariances=np.eye(7)[None]*.04,
        weights=np.ones(1),classes=np.full((1,3),1/3),binary_reference_fraction=.17,binary_brightness_factor=2.)
    raw=np.array([.2,.3,.25]);trans=np.array([.9,.95,.97]);cal=np.array([.01,-.01,.02])
    means=np.array([11.]);sigmas=np.array([1.]);corrections=np.array([np.zeros((3,2)),[[.4,.3],[-.2,.8],[.1,.2]]])
    error,lower=candidate_thresholds(raw,trans,cal);expected=[]
    for correction in corrections:
        m,s=transformed_radial_parameters(means,sigmas,correction);rows=[]
        for c in range(3):
            answer=0.
            for fraction,factor in [(.83,1.),(.17,2.)]:
                def f(mu):
                    a=10**(-.4*(mu-10))*factor
                    return norm.pdf(mu,m[c,0],s[c,0])*ndtr((20*a-lower)/np.sqrt(.04*a*a+error**2)).prod()
                answer+=fraction*quad(f,-5,30,epsabs=1e-10,points=[10,11,12,13])[0]
            rows.append([answer])
        expected.append(rows)
    actual,audit=shared_gaussian_entry(model,raw,trans,cal,means,sigmas,corrections,candidate=True,
        support=lambda x:np.ones(len(x),bool),draws=131072)
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=0)
    assert audit['pruning_bound'].max()<1e-13

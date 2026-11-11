import numpy as np
from scipy.integrate import quad
from scipy.stats import norm,lognorm
from v2_6_luminosity import signed_loglike


def test_signed_parallax_matches_independent_convolution():
    for mu,sigma,p,error in [(12,.3,.4,.01),(12,.5,.4,.01),(12,.5,-.2,.2),(12,.1,.5,.1)]:
        s=sigma*np.log(10)/5;scale=10**((10-mu)/5)
        # Direct positive true-parallax integration, independent of both GH paths.
        expected=quad(lambda x:lognorm.pdf(x,s,scale=scale)*norm.pdf(p,x,error),0,3,epsabs=1e-10,points=[scale,max(p,.01)])[0]
        got=signed_loglike(np.array([mu]),np.array([sigma]),np.array([p]),np.array([error]))[0]
        np.testing.assert_allclose(got,np.log(expected),atol=2e-4)


def test_zero_scatter_and_offset_sign():
    mu=np.array([12.]);p=np.array([.5]);e=np.array([.05])
    np.testing.assert_allclose(signed_loglike(mu,np.zeros(1),p,e),norm.logpdf(p,10**((10-mu)/5),e))
    assert signed_loglike(mu,np.zeros(1),p,e,offset=.5)>signed_loglike(mu,np.zeros(1),p,e,offset=-.5)

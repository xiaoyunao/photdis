import numpy as np
from scipy.stats import norm
from v2_3_diagnostics import teacher_log, interval95


def test_convolution_uniform_and_extreme():
    edges=np.array([0.,1.]);pdf=np.ones((3,1));target=np.array([.5,2.,100.]);sig=np.array([.2,.3,.1])
    actual=teacher_log(pdf,edges,target,sig)
    expected=np.log(norm.cdf((1-target[:2])/sig[:2])-norm.cdf(-target[:2]/sig[:2]))
    np.testing.assert_allclose(actual[:2],expected,rtol=1e-9)
    assert np.isfinite(actual[2]) and actual[2]< -100000


def test_interval95_uses_outer_quantiles():
    mu=np.array([[0.,1.,2.,3.,4.]])
    assert interval95(mu,np.array([2.]))[0]==4
    assert interval95(mu,np.array([5.]))[0]==44

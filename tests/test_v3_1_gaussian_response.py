import numpy as np
from scipy.stats import norm
from photdis.v3_1.gaussian_response import class_gaussian_logresponse
from photdis.v3_1.gaussian_entry import transformed_radial_parameters


def test_native_gaussian_response_matches_analytic_integral_and_preserves_log_tails():
    grid=np.linspace(-20,50,35001);center=np.array([11.,15.,18.]);width=np.array([.07,.3,.2])
    loglike=norm.logpdf(grid[None],center[:,None],width[:,None])-20000
    means=np.array([8.,14.,20.5]);sigmas=np.array([3.,1.4,3.5])
    correction=np.array([np.zeros((3,2)),[[.4,.3],[-.2,.8],[0.,.1]]])
    expected=[]
    for p in correction:
        m,s=transformed_radial_parameters(means,sigmas,p)
        expected.append(norm.logpdf(center[:,None],m,np.sqrt(s*s+width[:,None]**2))-20000)
    actual=class_gaussian_logresponse(loglike,grid,means,sigmas,correction)
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.gaussian_response import class_gaussian_logresponse
from photdis.v3_1.gaussian_joint import interval_linear_weights,normal_log_bin_probability,class_gaussian_joint_logresponse


def test_piecewise_bin_weights_conserve_nonuniform_trapezoids():
    grid=np.array([-5.,-1.1,0.,.4,2.,9.]);edges=np.array([-7.,-.8,.3,8.,12.])
    weights=interval_linear_weights(grid,edges)
    assert (weights>=0).all()
    np.testing.assert_allclose(weights.sum(axis=0),trapezoid_weights(grid),atol=2e-15)
    # Integrating a constant over each clipped interval is exact.
    expected=np.diff(np.clip(np.r_[-np.inf,edges,np.inf],grid[0],grid[-1]))
    np.testing.assert_allclose(weights.sum(axis=1),expected,atol=2e-15)


def test_conditional_bin_tails_remain_log_finite_and_normalized():
    values=normal_log_bin_probability(np.array([-30.,0.,30.]),.3,[-1.,1.])
    assert np.isfinite(values).all()
    np.testing.assert_allclose(logsumexp(values,axis=0),0.,atol=1e-14)
    assert values.min()<-4000


def test_true_distance_bins_match_analytic_convolution_posterior():
    grid=np.linspace(-10,40,25001);center=np.array([10.,15.,20.]);noise=np.array([.2,.4,.1])
    native=norm.logpdf(grid[None],center[:,None],noise[:,None])-20000
    means=np.array([12.,19.]);sigmas=np.array([2.,3.]);boundaries=np.array([11.,14.,17.,22.])
    corrections=np.array([np.zeros((3,2)),[[.7,.3],[-.4,.9],[.2,.6]]])
    actual=class_gaussian_joint_logresponse(native,grid,means,sigmas,corrections,boundaries)
    total=class_gaussian_logresponse(native,grid,means,sigmas,corrections)
    np.testing.assert_allclose(logsumexp(actual,axis=-1),total,atol=1e-10,rtol=0)
    for j,p in enumerate(corrections):
        for c in range(3):
            for d in range(2):
                variance=noise[c]**2+p[c,1]**2
                posterior_variance=1/(1/sigmas[d]**2+1/variance)
                mean=posterior_variance*(means[d]/sigmas[d]**2+(center[c]+p[c,0])/variance)
                expected=normal_log_bin_probability(np.array([mean]),np.sqrt(posterior_variance),boundaries)[:,0]
                # Zero-scatter bins integrate a piecewise-linear density at finite grid spacing.
                np.testing.assert_allclose(np.exp(actual[j,c,d]-total[j,c,d]),np.exp(expected),atol=3e-6,rtol=0)

import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm
from photdis.cluster_rgb_scales import ccm_r, change_scale, reference_covariance
from photdis.cluster_rgb_shared_distance import (
    member_log_likelihood, log_normal_interval, marginalize_members,
    prior_member_log_likelihood, posterior_summary)


def test_physical_shift_changes_color_label_and_reference_together():
    g,c,m = change_scale(np.array([18.]),np.array([[.8,.4]]),
                         np.array([[.06,.04,.02]]),np.array([.03,.02,.01]),16.,.1)
    np.testing.assert_allclose(g,[18.03])
    np.testing.assert_allclose(c,[[.81,.41]])
    np.testing.assert_allclose(m,[1.93])
    assert abs(ccm_r(np.array([1e4/1.82]))[0]-3.1) < 1e-12


def test_reference_correlation_does_not_duplicate_error_budget():
    independent = reference_covariance([.017,.024],[.036,.033],0)
    shared = reference_covariance([.017,.024],[.036,.033],1)
    np.testing.assert_allclose(np.diag(independent),np.diag(shared))
    d = np.array([1.,-1.])
    assert d@shared@d < d@independent@d


def test_analytic_segment_integral_matches_direct_multivariate_quadrature():
    model = dict(kind='monotone',beta=[[.9,.825,.75,.65,.12],[.5,.44,.38,.30,.06]],
                 intrinsic=[[.0004,.0001],[.0001,.0003]])
    from photdis.cluster_rgb_lower import mean
    g = np.array([17.4]); c = np.array([[.78,.42]]); err = np.array([[.02,.03,.04]])
    mus = np.array([14.6,15.5,16.95])
    actual = member_log_likelihood(model,mus,g,c,err,-1.7)[0]
    t = np.array([[1,0,0],[1,-1,0],[0,1,-1]])
    cov = t@np.diag(err[0]**2)@t.T
    cov[1:,1:] += model['intrinsic']
    for j,mu in enumerate(mus):
        # Split at knots and around the narrow g likelihood peak.
        points = sorted(set([.5,1.25,2,3]+[float(x) for x in g[0]-mu+np.arange(-5,6)*.02 if .5<x<3]))
        y = np.r_[g[0]-mu,c[0]]
        f = lambda m: multivariate_normal.pdf(y,mean=np.r_[m,mean(model,np.array([m]),-1.7)[0]],cov=cov)/2.5
        expected = sum(quad(f,a,b,epsabs=1e-13)[0] for a,b in zip(points[:-1],points[1:]))
        np.testing.assert_allclose(np.exp(actual[j]),expected,rtol=1e-7,atol=1e-12)


def test_shared_offset_floor_and_wrong_order():
    # Exact Gaussian model: shared zero tau=.2, independent error sigma=.1.
    eta = np.linspace(-1,1,1001); mu = np.linspace(-.8,.8,801)
    # Equal-weight normal quantiles integrate its prior directly.
    eta = norm.ppf((np.arange(1001)+.5)/1001)*.2
    ll = norm.logpdf(mu[None,None,:]+eta[:,None,None],scale=.1)
    for n in (1,20):
        right, wrong = marginalize_members(np.repeat(ll,n,axis=1))
        variance = lambda q: np.sum(mu**2*np.exp(q-np.logaddexp.reduce(q)))
        np.testing.assert_allclose(variance(right),.2**2+.1**2/n,rtol=.01)
        np.testing.assert_allclose(variance(wrong),(.2**2+.1**2)/n,rtol=.01)


def test_normal_interval_far_tails():
    np.testing.assert_allclose(log_normal_interval(9.,10.),log_normal_interval(-10.,-9.))
    assert np.isfinite(log_normal_interval(100.,101.))


def test_known_luminosity_endpoints_can_constrain_distance_without_colors():
    grid = np.arange(15.9,16.1,.0001)
    # Known Mg interval and two endpoint members alone imply mu in [15.99,16.01].
    ll = prior_member_log_likelihood(grid,np.array([16.51,18.99]),np.full(2,.001))
    result,_ = posterior_summary(grid,ll.sum(axis=0))
    assert abs(result['q'][2]-16) < .001
    assert result['width68'] < .02

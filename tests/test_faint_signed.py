import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from photdis.faint_signed import (
    K, components, scale_weights, selected_draw, selected_moments, selected_score,
)


def test_conditional_pdf_and_moments_against_direct_integration():
    edges = np.array([[9.0, 11.0, 13.0, 15.0]])
    mass = np.array([[.15, .55, .30]])
    y, sigma, limit = .06, .2, .34
    p,w = components(edges,mass,40)

    def integral(func):
        return sum(weight/(b-a)*quad(lambda x: func(np.exp(K*(10-x-.1))+.015),a,b)[0]
                   for a,b,weight in zip(edges[0,:-1],edges[0,1:],mass[0]))

    selection = integral(lambda mean: norm.cdf(limit,mean,sigma))
    result = selected_score(p,w,np.array([y]),np.array([sigma]),np.array([limit]),.1,.015)
    expected = [
        np.log(integral(lambda mean: norm.pdf(y,mean,sigma))/selection),
        integral(lambda mean: norm.cdf(y,mean,sigma))/selection,
        integral(lambda mean: norm.cdf(0,mean,sigma))/selection,
    ]
    np.testing.assert_allclose(np.asarray(result).ravel(),expected,atol=2e-12)
    m,v,q = selected_moments(p,w,np.array([sigma]),np.array([limit]),.1,.015)
    first = quad(lambda x: x*integral(lambda mean: norm.pdf(x,mean,sigma)),
                 -2,limit,epsabs=1e-11)[0]/selection
    second = quad(lambda x: x*x*integral(lambda mean: norm.pdf(x,mean,sigma)),
                  -2,limit,epsabs=1e-11)[0]/selection
    np.testing.assert_allclose([m.item(),v.item(),q.item()],[first,second-first**2,selection],atol=2e-12)


def test_selected_draw_respects_the_cut_and_keeps_negative_values():
    sigma = np.array([.12,.23])
    lim = 2*sigma
    mass = np.array([[.65,.35],[.4,.6]])
    cdf = np.cumsum(mass,axis=1)
    edges = np.array([[10,12,14],[11,13,15]],float)
    offsets = np.full((len(sigma),24000),.025)
    sample = selected_draw(edges,cdf,sigma,lim,.2,offsets,np.random.default_rng(71))
    assert (sample < lim[:,None]).all()
    assert np.count_nonzero(sample<0) > 1000
    p,w = components(edges,mass,32)
    ex,var,_ = selected_moments(p,w,sigma,lim,.2,.025)
    np.testing.assert_allclose(sample.mean(axis=1),ex,atol=.003)
    np.testing.assert_allclose(sample.var(axis=1),var,atol=.001)


def test_scale_projection_matches_dense_spatial_covariance():
    block = np.array([0,0,1,2,2])
    slope = np.array([-.1,-.12,-.09,-.10,-.07])
    h = np.array([1,.98,.95,.9,1])
    var = np.array([.032,.08,.05,.03,.10])
    field_rms,block_rms = .01,.02
    covariance = np.diag(var) + np.outer(h,h)*(field_rms**2+block_rms**2*(block[:,None]==block))
    v = np.linalg.solve(covariance,slope)
    result = scale_weights(slope,h,var,block,field_rms,block_rms)
    np.testing.assert_allclose(result,v/np.dot(v,slope),atol=2e-14)
    np.testing.assert_allclose(np.dot(result,slope),1,atol=2e-14)

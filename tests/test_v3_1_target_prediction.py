import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from photdis.v3_1.target_prediction import marginalize_class_population
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3.likelihood import trapezoid_weights


def test_conditional_transfer_keeps_each_population_normalizer_and_deep_log_tails():
    g=np.linspace(-20,50,7001);ll=np.array([norm.logpdf(g,m,s)-20000 for m,s in [(10,.5),(15,.6),(18,.7)]])
    ll-=np.array([0.,2.,5.])[:,None]
    theta=np.array([np.arange(1,16).reshape(3,5),np.arange(15,0,-1).reshape(3,5)],float);theta/=theta.sum(axis=(1,2),keepdims=True)
    w=np.array([.25,.75]);r=marginalize_class_population(ll,g,theta,w)
    radial=norm.pdf(g[None],DISTANCE_BASIS_MEAN[:,None],DISTANCE_BASIS_SIGMA[:,None]);tw=trapezoid_weights(g);expected=[]
    for t in theta:
        logjoint=ll+np.log(t@radial)
        logjoint-=logsumexp(logjoint+np.log(tw)[None]);expected.append(logjoint)
    exact=logsumexp(np.array(expected)+np.log(w)[:,None,None],axis=0)
    np.testing.assert_allclose(r['class_logpdf'],exact,atol=2e-11,rtol=0)
    assert abs(r['log_normalization'])<1e-10 and r['joint_delta']<1e-10
    np.testing.assert_allclose(r['draw_probability'].sum(axis=(1,2)),1,atol=1e-10)
    assert np.isfinite(r['logpdf']).all() and r['logpdf'].min() < -1000
    plugged=marginalize_class_population(ll,g,(w[:,None,None]*theta).sum(axis=0)[None],np.ones(1))
    assert abs(plugged['class_distance']-r['class_distance']).max()>.001

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from photdis.cluster_rgb_local_color import (
    brightness_log_marginal,conditional_mass,summarize_local,expand_abundance_atoms,
    LocalColorTarget,component_bin_mass,weighted_joint)
from photdis.cluster_rgb_lower import mean
from photdis.cluster_rgb_shared_sampling import effective_covariance

MODEL=dict(kind='monotone',beta=[[.9,.825,.75,.65,.12],[.5,.44,.38,.30,.06]],
    intrinsic=[[.0004,.0001],[.0001,.0003]],contamination={'weights':[.95,.05],'scales':[1.,25.]})


def test_ratio_after_marginalization_and_no_color_identity():
    # Shared-state weighting by brightness changes the conditional experiment.
    lg=np.log([[.1,.8],[.7,.2]]);lc=np.log([[.9,.2],[.1,.8]])
    correct=logsumexp(lg+lc,axis=0)-logsumexp(lg,axis=0)
    wrong=logsumexp(lc,axis=0)-np.log(2)
    assert not np.allclose(correct,wrong)
    np.testing.assert_allclose(np.exp(correct),[.2,.32])
    edges=np.linspace(-.5,.5,201);b=-25*np.linspace(-1,1,200)**2
    no_color=conditional_mass(np.exp(b-b.max()),b)
    np.testing.assert_allclose(no_color,1/200)
    stats=summarize_local(edges,no_color)
    np.testing.assert_allclose([stats['q'][2],stats['width68'],stats['width95'],stats['kl_nats']],
                               [0,.68,.95,0],atol=1e-12)


def test_brightness_shared_integral_matches_adaptive_quadrature():
    from scipy.special import ndtr
    mu=np.array([15.,15.3]);g=np.array([15.65,17.7]);sg=np.array([.05,.08]);tau=.07
    got=brightness_log_marginal(mu,g,sg,tau,128)
    expected=[]
    for m in mu:
        def f(t):
            intrinsic=g+t*tau-m
            p=ndtr((3-intrinsic)/sg)-ndtr((.5-intrinsic)/sg)
            return np.prod(p/2.5)*np.exp(-t*t/2)/np.sqrt(2*np.pi)
        expected.append(np.log(quad(f,-10,10,epsabs=1e-12)[0]))
    np.testing.assert_allclose(got,expected,atol=1e-9)


def test_abundance_is_shared_atom_and_model_jump_is_reversible():
    models,labels=expand_abundance_atoms([MODEL],[-1.46,-2.])
    for m,z in zip(models,[-1.46,-2.]):
        np.testing.assert_allclose(mean(m,np.array([1.,2.]),-1.7),mean(MODEL,np.array([1.,2.]),z))
    cov=effective_covariance(MODEL,np.array([3.57,2.58,1.45]),.005,.01,0.)
    target=LocalColorTarget(models,np.array([17.,18.]),np.array([[.8,.4],[.7,.3]]),
        np.full((2,3),.02),-1.7,[cov,cov],2,(15.5,16.5))
    x=np.array([16.,.01,-.02]);y=target.model_proposal(x,0,1)
    np.testing.assert_allclose(target.model_proposal(y,1,0),x,atol=1e-14)
    eps=1e-6;jac=np.column_stack([(target.model_proposal(x+eps*v,0,1)-y)/eps for v in np.eye(3)])
    np.testing.assert_allclose(np.linalg.det(jac),1.,atol=1e-8)


def test_component_bins_keep_evidence_and_are_stable_in_truncated_tails():
    edges=np.linspace(-.5,.5,201)
    mass=component_bin_mass(edges,np.array([0.,0.,0.,.4]),np.full(4,.03),np.array([0,0,0,1]),2)
    np.testing.assert_allclose(mass.sum(axis=1),[.75,.25],atol=1e-12)
    p=weighted_joint(mass,[.8,.2])
    np.testing.assert_allclose(p[edges[:-1]>.25].sum(),.05/.65,atol=1e-5)
    # Mean lies far below truncation; ordinary CDF subtraction would underflow.
    tail=component_bin_mass(edges,np.array([-2.]),np.array([.02]),np.array([0]),1)
    assert np.all(np.isfinite(tail))
    np.testing.assert_allclose(tail.sum(),1.,atol=1e-10)

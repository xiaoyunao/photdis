import numpy as np
from photdis.cluster_rgb_lower import mean
from photdis.cluster_rgb_shared_distance import member_log_likelihood
from photdis.cluster_rgb_shared_sampling import (
    PreparedLikelihood,effective_covariance,chain_diagnostics,metropolis_chains,conditional_density)


MODEL=dict(kind='monotone',beta=[[.9,.825,.75,.65,.12],[.5,.44,.38,.30,.06]],
           intrinsic=[[.0004,.0001],[.0001,.0003]],
           contamination={'weights':[.95,.05],'scales':[1.,25.]})


def test_collapsed_physical_nuisance_is_exact_and_cache_agrees():
    g=np.array([16.8,17.5]);colors=np.array([[.84,.45],[.72,.34]])
    errors=np.array([[.02,.03,.04],[.01,.02,.03]])
    feh=-1.7;dz=.15;dust=.006;zero=np.array([.01,-.006,.002]);r=np.array([3.57,2.58,1.45])
    shift=zero-r*dust;photo_color=np.array([shift[0]-shift[1],shift[1]-shift[2]])
    dc=photo_color-np.asarray(MODEL['beta'])[:,-1]*dz
    original=member_log_likelihood(MODEL,np.array([16.]),g+shift[0],colors+photo_color,errors,feh+dz)[:,0]
    prepared=PreparedLikelihood(MODEL,g,colors,errors,feh)
    collapsed=prepared.rows(16.-shift[0],dc)
    np.testing.assert_allclose(collapsed,original,atol=1e-10)


def test_effective_covariance_matches_original_gaussian_draws():
    rng=np.random.default_rng(12);r=np.array([3.57,2.58,1.45]);n=200000
    shift=rng.normal(0,.01,(n,3))-rng.normal(0,.005,(n,1))*r
    c=np.column_stack([shift[:,0],shift[:,0]-shift[:,1],shift[:,1]-shift[:,2]])
    c[:,1:]-=rng.normal(0,.2,(n,1))*np.asarray(MODEL['beta'])[:,-1]
    np.testing.assert_allclose(np.cov(c,rowvar=False),effective_covariance(MODEL,r,.005,.01,.2),rtol=.02,atol=2e-6)


def test_diagnostics_detect_chain_location_and_scale_disagreement():
    rng=np.random.default_rng(7);x=rng.normal(size=(4,2000))
    d=chain_diagnostics(x)
    assert d['rhat']<1.01 and d['bulk_ess']>4000
    x[0]+=2
    assert chain_diagnostics(x)['rhat']>1.1
    x[0]-=2;x[0]*=.1
    assert chain_diagnostics(x)['rhat']>1.1


def test_fixed_postwarmup_sampler_recovers_correlated_gaussian():
    cov=np.array([[.04,.003],[.003,.0004]]);inv=np.linalg.inv(cov)
    class Target:
        def __call__(self,x,k): return -.5*x@inv@x
        def model_proposal(self,x,old,new):return x.copy()
    x,k,lp,accept=metropolis_chains(Target(),[0,0],1,chains=4,warmup=1600,draws=5000,seed=11)
    np.testing.assert_allclose(np.cov(x.reshape(-1,2),rowvar=False),cov,rtol=.08,atol=.0002)
    assert chain_diagnostics(x[:,:,0])['rhat']<1.02
    # Marginalized shared Gaussian width survives arbitrarily many identical means.
    grid=np.arange(14,18,.002)
    p=conditional_density(grid,np.full(100,16.),np.full(100,.2),bounds=(14,18))
    np.testing.assert_allclose(np.sum((grid-16.)**2*p),.04,atol=1e-5)


def test_curve_following_proposal_is_reversible_with_unit_jacobian():
    from photdis.cluster_rgb_shared_sampling import CollapsedTarget
    cov=effective_covariance(MODEL,np.array([3.57,2.58,1.45]),.005,.01,.2)
    target=CollapsedTarget([MODEL],np.array([17.,18.]),np.array([[.8,.4],[.7,.3]]),
                           np.full((2,3),.02),-1.7,[cov],2)
    x=np.array([16.,.01,-.02]);shift=.35
    y=target.distance_proposal(x,shift,0)
    np.testing.assert_allclose(target.distance_proposal(y,-shift,0),x,atol=1e-14)
    eps=1e-6
    jac=np.column_stack([(target.distance_proposal(x+eps*v,shift,0)-y)/eps for v in np.eye(3)])
    np.testing.assert_allclose(np.linalg.det(jac),1.,atol=1e-8)

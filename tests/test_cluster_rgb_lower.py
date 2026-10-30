import numpy as np
from photdis.cluster_rgb_lower import design, mean, slope, fit, probabilities, summarize


def test_monotone_fit_cannot_reverse_at_faint_end():
    m=np.linspace(.5,2.99,120)
    z=np.repeat([-2.,-1.46],60)
    y=.7-.09*m+.03*(z+1.7)+.005*np.sin(m*20)
    a=dict(mg=m,feh=z,color=np.column_stack([y,.5*y]),
           sg=np.full(120,.003),sr=np.full(120,.003),sz=np.full(120,.003),
           cluster=np.repeat(['a','b'],60))
    mod=fit(a,2,'monotone')
    assert np.max(slope(mod,np.linspace(.5,3,400)))<1e-7
    assert np.max(abs(mean(mod,m,z)[:,0]-y))<.02


def test_piecewise_derivative_matches_finite_difference():
    model=dict(kind='monotone',beta=[[.7,.6,.5,.4,.03],[.4,.35,.3,.2,.02]])
    m=np.array([.6,1.4,2.5])
    numeric=(mean(model,m+1e-6,-1.7)-mean(model,m-1e-6,-1.7))/2e-6
    np.testing.assert_allclose(slope(model,m),numeric,atol=1e-9)


def test_uniform_baseline_and_distance_sign():
    grid=np.arange(.51,3,.02)
    prob=np.full((2,len(grid)),1/len(grid))
    stats,q,r=summarize(grid,prob,[1.,2.5])
    np.testing.assert_allclose(r,[-.75,.75],atol=.021)
    assert abs(stats['overall']['log_density']+np.log(2.5))<1e-10
    assert abs(stats['overall']['width68']-1.70)<.021


def test_batch_posterior_matches_frozen_v1():
    from photdis.cluster_rgb_conditional import posterior_on_grid
    model=dict(kind='quadratic',beta=[[.6,-.1,.001,.04],[.3,-.05,.001,.02]],
               intrinsic=[[.0004,.0001],[.0001,.0003]])
    grid=np.arange(.51,3,.02)
    p=probabilities([model],grid,[-1.7],[1],[[.5,.25]],[[.01,.02,.03]])[0]
    expected=posterior_on_grid([model],grid,[-1.7],[1],[.5,.25],.01,.02,.03)
    np.testing.assert_allclose(p,expected,atol=1e-12)


def test_robust_mixture_keeps_outlier_with_finite_likelihood():
    from photdis.cluster_rgb_lower import fit_contamination
    rng=np.random.default_rng(38)
    m=np.tile(np.linspace(.51,2.99,50),2)
    z=np.repeat([-2.,-1.46],50)
    y=.7-.08*m+.025*(z+1.7)+rng.normal(0,.01,len(m));y[5]+=.3
    a=dict(mg=m,feh=z,color=y[:,None],sg=np.full(100,.003),sr=np.full(100,.003),
           sz=np.full(100,.003),cluster=np.repeat(['a','b'],50))
    mod=fit_contamination(a,1)
    assert mod['contamination']['weights']==[.95,.05]
    assert np.max(slope(mod,np.linspace(.5,3,50)))<1e-7
    assert abs(mean(mod,np.array([1.5]),-1.7)[0,0]-(.7-.08*1.5))<.03
    p=probabilities([mod],np.arange(.51,3,.02),[-1.7],[1],[[1.0]],[[.003]*3])
    np.testing.assert_allclose(p.sum(),1)

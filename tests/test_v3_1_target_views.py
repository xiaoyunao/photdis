import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
import v3_1_validation_target as driver
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA


def test_final_corrected_views_match_analytic_population_mixture(monkeypatch):
    grid=np.linspace(-20,50,7001);location=np.array([8.,13.,18.]);width=np.array([.7,1.,1.2])
    native=np.array([norm.logpdf(grid,m,s) for m,s in zip(location,width)])
    correction=[np.zeros((3,2)),np.tile([.2,.3],(3,1)),np.tile([-.4,.2],(3,1))]
    theta={};analytic=[]
    for state,parameter in enumerate(correction):
        raw=np.arange(1,31,dtype=float).reshape(2,3,5)
        raw=np.roll(raw,state,axis=1);raw[0]*=np.array([4.,1.,.3])[:,None]
        raw/=raw.sum(axis=(1,2),keepdims=True);theta[state]={7:raw};draws=[]
        mean=location+parameter[:,0];variance=width**2+parameter[:,1]**2
        pv=DISTANCE_BASIS_SIGMA**2;pm=DISTANCE_BASIS_MEAN
        postvar=1/(1/variance[:,None]+1/pv[None]);postmean=postvar*(mean[:,None]/variance[:,None]+pm[None]/pv[None])
        evidence=norm.logpdf(mean[:,None],pm[None],np.sqrt(variance[:,None]+pv[None]))
        for t in raw:
            lw=np.log(t)+evidence;lw-=logsumexp(lw)
            draws.append(logsumexp(lw[:,:,None]+norm.logpdf(grid[None,None],postmean[:,:,None],np.sqrt(postvar[:,:,None])),axis=(0,1)))
        analytic.append(logsumexp(draws,axis=0)-np.log(2))
    views={'zero':dict(ids=np.array([0]),log_weights=np.array([0.]),accepted=True),
        'mixed':dict(ids=np.array([1,2]),log_weights=np.log([.2,.8]),accepted=True)}
    monkeypatch.setattr(driver,'_STATES',[dict(parameters=p.tolist()) for p in correction]);monkeypatch.setattr(driver,'_THETA',theta);monkeypatch.setattr(driver,'_VIEWS',views)
    arrays,results,_=driver.predict(native,grid,7)
    for name,view in views.items():
        expected=logsumexp(np.array(analytic)[view['ids']]+view['log_weights'][:,None],axis=0)
        selected=expected>expected.max()-50
        # The production operator has registered 8-sigma convolution support;
        # the analytic reference integrates unbounded Gaussian scatter. Its
        # relative tail effect reaches 6.1e-8 at 50 log-density units here.
        np.testing.assert_allclose(arrays[name+'_logpdf'][selected],expected[selected],atol=1e-7,rtol=0)
        assert results[name]['joint_delta']<1e-10

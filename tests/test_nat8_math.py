"""Independent checks of calibration, shared distance and full-distribution scores."""
import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.optimize import approx_fprime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nat8_math import objective,fit,fast_quantiles
from nat8_statsmath import hist_curves,curve_values,direct_queries


def test_gradient_and_common_distance_translation():
    means=np.array([[14.,14.7],[14.2,15.]]);lw=np.log([[.4,.6],[.7,.3]]);sigma=np.array([.15,.2]);x=np.array([-.1,.3]);counts=np.array([2.,1.]);a=np.array([-.3,.7]);mu=14.5
    f,g=objective(a,means,lw,sigma,x,mu,counts);numeric=approx_fprime(a,lambda aa:objective(aa,means,lw,sigma,x,mu,counts)[0],1e-7)
    np.testing.assert_allclose(g,numeric,atol=8e-6)
    f2,g2=objective(a-[.07,0],means,lw,sigma,x,mu+.07,counts)
    np.testing.assert_allclose([f,*g],[f2,*g2],atol=1e-11)


def test_fit_recovers_fixed_two_parameter_map():
    x=np.linspace(-.15,.6,10);a=np.array([-.4,1.1]);mu=14.4;means=(mu+a[0]+a[1]*x)[:,None];lw=np.zeros_like(means)
    # Each star's single active node occupies its own color; off-diagonal
    # alternatives have exactly zero weight, rather than a luminosity fit.
    mm=np.broadcast_to(means,(10,10));weights=np.full((10,10),-np.inf);np.fill_diagonal(weights,0)
    result=fit(mm,weights,np.full(10,.1),x,mu,np.ones(10),[[0,0]])
    np.testing.assert_allclose(result['a'],a,atol=1e-7)


def test_shared_reference_cancels_in_calibrated_M5_queries():
    m=np.array([14.,15.]);lw=np.log([.4,.6]);x=np.array([.1,.4]);a=np.array([-.2,.7]);shift=np.array([-.05,0,.08]);aa=np.tile(a,(3,1));aa[:,0]-=shift
    p,l=direct_queries(m,lw,.1,x,aa,(14.5+shift)[:,None]);np.testing.assert_allclose(p,np.broadcast_to(p[0],p.shape),atol=1e-13);np.testing.assert_allclose(l,np.broadcast_to(l[0],l.shape),atol=1e-12)


def test_histogram_energy_score_and_quantiles_against_gaussian():
    x,cdf,crps,pdf=hist_curves(np.array([0.]),np.array([1.]),1.,.0025,query=np.array([0.]))
    expected=(np.sqrt(2)-1)/np.sqrt(np.pi);np.testing.assert_allclose(curve_values(x,crps,[[0]]),expected,atol=1e-6)
    q=fast_quantiles(np.array([0.]),np.array([0.]),1.)
    np.testing.assert_allclose(ndtr(q),[.025,.16,.5,.84,.975],atol=1e-9)
    assert abs(cdf[0,-1]-1)<1e-10


def test_full_log_tail_survives_ordinary_weight_underflow():
    means=np.array([0.,100.]);lw=np.array([0.,-1000.]);a=np.zeros(2);x=np.zeros(2);y=np.array([[100.]])
    p,l=direct_queries(means,lw,.1,x,a,y)
    np.testing.assert_allclose(l[0,0],-1000-np.log(.1*np.sqrt(2*np.pi)),atol=1e-10)

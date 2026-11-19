"""Independent identities for the fixed sequence-mechanism probability math."""
import sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.integrate import trapezoid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from seq7_math import imf_integral,mixture,evaluate_mix,quantiles,uniform_curve,curve_crps,row_interp


def test_mass_measure_is_subdivision_invariant():
    edges=np.linspace(.3,.9,101)
    np.testing.assert_allclose(imf_integral(edges[:-1],edges[1:]).sum(),imf_integral(.3,.9),rtol=1e-13)
    np.testing.assert_allclose(imf_integral(.6,.8),.5*(.8**(-1.3)-.6**(-1.3))/(-1.3),rtol=1e-13)


def test_gray_shift_changes_only_distance():
    m=np.array([[18.,17.5,17.2]]);e=np.full_like(m,.02);nodes=np.array([[5.,4.5,4.2],[4.,3.5,3.2]])
    a=mixture(m,e,nodes,np.array([.25,.75]));b=mixture(m+2,e,nodes,np.array([.25,.75]))
    np.testing.assert_allclose(b[0]-a[0],2);np.testing.assert_allclose(b[1],a[1]);np.testing.assert_allclose(quantiles(b[0],b[1],b[2])-quantiles(a[0],a[1],a[2]),2)


def test_reference_convolution_retains_tiny_tails():
    x=np.linspace(-2,2,10001);cdf,lp=uniform_curve(x,np.array([-.5,.5]),np.array([1.]),.1)
    np.testing.assert_allclose(trapezoid(np.exp(lp),x),1,atol=1e-10)
    np.testing.assert_allclose(cdf[len(x)//2],.5,atol=1e-13)
    _,tails=uniform_curve(np.array([-100.,100.]),np.array([-.5,.5]),np.array([1.]),.1)
    assert np.isfinite(tails).all() and (tails<-1e5).all()


def test_crps_gaussian_identity_and_row_queries():
    x=np.arange(-9,9.0001,.0025);cdf=ndtr(x);v=curve_crps(x,cdf);expected=2/np.sqrt(2*np.pi)-1/np.sqrt(np.pi)
    np.testing.assert_allclose(np.interp(0,x,v),expected,atol=1e-6)
    values=np.array([x,x*2]);query=np.array([[.1,.2],[.3,.4]])
    np.testing.assert_allclose(row_interp(x,query,values),[[.1,.2],[.6,.8]],atol=1e-10)

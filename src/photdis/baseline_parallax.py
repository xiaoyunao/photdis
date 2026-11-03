"""Observed-parallax checks of histogram distance predictions, including selection."""
import numpy as np
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss


def normal_interval(a,b):
    """Stable standard-normal mass for ordered endpoints (including infinity)."""
    return np.maximum(0.,np.where(a>0,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a)))


def parallax_check(pdf,edges,g,observed,error,snr_lo,snr_hi,order=8):
    """CDF and moments conditional on the *observed* S/N selection.

    Histogram entries are bin masses, uniform within M_effective bins. A shared
    g perturbation is already encoded in M_effective and must not be added again.
    Results describe each empirical prediction, not a calibrated population prior.
    """
    node,weight=leggauss(order);weight=weight/2
    absolute=(edges[:-1,None]+edges[1:,None])/2+np.diff(edges)[:,None]*node/2
    n=len(g);cdf=np.zeros(n);prob=np.zeros(n);mean=np.zeros(n);second=np.zeros(n)
    density=np.zeros(n)
    for j in range(order):
        par=10**(2-(g[:,None]-absolute[:,j])/5)
        sig=error[:,None];a=snr_lo[:,None]-par/sig;b=snr_hi[:,None]-par/sig
        z=(observed[:,None]-par)/sig
        mass=normal_interval(a,b);p=pdf*weight[j]
        prob+=np.sum(p*mass,axis=1)
        cdf+=np.sum(p*normal_interval(a,z),axis=1)
        phi_a=np.exp(-.5*a*a)/np.sqrt(2*np.pi);phi_b=np.exp(-.5*b*b)/np.sqrt(2*np.pi)
        first=par*mass+sig*(phi_a-phi_b)
        ap=np.zeros_like(a);bp=np.zeros_like(b)
        np.multiply(a,phi_a,out=ap,where=np.isfinite(a));np.multiply(b,phi_b,out=bp,where=np.isfinite(b))
        raw2=(par**2+sig**2)*mass+2*par*sig*(phi_a-phi_b)+sig**2*(ap-bp)
        mean+=np.sum(p*first,axis=1);second+=np.sum(p*raw2,axis=1)
        density+=np.sum(p*np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*sig),axis=1)
    valid=prob>1e-280
    pit=np.divide(cdf,prob,out=np.full(n,np.nan),where=valid)
    expectation=np.divide(mean,prob,out=np.full(n,np.nan),where=valid)
    variance=np.divide(second,prob,out=np.full(n,np.nan),where=valid)-expectation**2
    return dict(pit=np.clip(pit,0,1),selection_probability=prob,conditional_mean=expectation,
                conditional_variance=np.maximum(variance,0),conditional_density=np.divide(density,prob,out=np.full(n,np.nan),where=valid))


def simulate_selected(pdf,edges,g,error,snr_lo,snr_hi,rng,repeats=256,order=8):
    """Draw the convolved mixture after exactly the observed-SNR selection."""
    from scipy.stats import truncnorm
    node,weight=leggauss(order);weight=weight/2
    absolute=(edges[:-1,None]+edges[1:,None])/2+np.diff(edges)[:,None]*node/2
    par=10**(2-(g[:,None,None]-absolute[None,:,:])/5)
    sig=error[:,None,None];a=snr_lo[:,None,None]-par/sig;b=snr_hi[:,None,None]-par/sig
    mass=pdf[:,:,None]*weight[None,None,:]*normal_interval(a,b)
    mass=mass.reshape(len(g),-1);par=par.reshape(len(g),-1);a=a.reshape(len(g),-1);b=b.reshape(len(g),-1)
    output=np.full((len(g),repeats),np.nan)
    for i in range(len(g)):
        total=mass[i].sum()
        if total<=1e-280:continue
        cdf=np.cumsum(mass[i]/total);cdf[-1]=1
        index=np.searchsorted(cdf,rng.random(repeats))
        output[i]=truncnorm.ppf(rng.random(repeats),a[i,index],b[i,index],loc=par[i,index],scale=error[i])
    return output

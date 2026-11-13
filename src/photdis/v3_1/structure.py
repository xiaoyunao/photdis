"""Shared-scale structure likelihood and an explicitly withdrawn Gaussian baseline."""
import numpy as np
from scipy.special import ndtri, roots_hermitenorm, roots_legendre
from scipy.stats import norm


def structure_quadrature(distance_order, odds_order):
    """Quadrature of the registered prior, with one common distance per catalog."""
    x,w=roots_legendre(distance_order)
    u=(x+1)/2
    d=9.03+np.where(u<.5,.56,.48)*ndtri(u)
    if (d<=0).any():raise ValueError('Nonpositive shared-scale quadrature distance')
    mu=10+5*np.log10(d)
    z,v=roots_hermitenorm(odds_order)
    return mu,np.log(w/2),4*z,np.log(v/np.sqrt(2*np.pi))


def withdrawn_gaussian_logresponse(log_evidence,posterior_mean,posterior_sigma,means,sigmas):
    """Integrate R_ref*N(mu;m,s)/N(mu;15,5) against each new Gaussian prior.

    The stable expression expands around the new prior mean, avoiding large
    subtracting squares. A nonintegrable likelihood approximation is rejected;
    neither the posterior variance nor the resulting evidence is clipped.
    """
    z,m,s=[np.asarray(a,float)[...,None] for a in (log_evidence,posterior_mean,posterior_sigma)]
    a,b=np.asarray(means,float),np.asarray(sigmas,float)
    if not all(np.isfinite(t).all() for t in (z,m,s,a,b)) or (s<=0).any() or (b<=0).any():
        raise ValueError('Finite moments and positive Gaussian widths required')
    precision=1/s**2+1/b**2-1/25.
    if (precision<=0).any():raise ValueError('Nonintegrable withdrawn Gaussian baseline')
    slope=(m-a)/s**2-(15-a)/25.
    return z+norm.logpdf(a,m,s)-norm.logpdf(a,15,5)-np.log(b)-.5*np.log(precision)+.5*slope*slope/precision


def selected_structure_loglike(log_structure,log_background,entry_structure,entry_background,radius,zeta):
    """One shared state; return catalog log likelihood and per-star membership.

    R contains photometric likelihood; A contains the same generative entry.
    Selection normalizes the catalog likelihood. Given actual selected flux,
    membership follows q*R/(q*R+(1-q)*R), not q*(R/A) renormalized per component.
    """
    rs,rb,as_,ab,r=map(lambda a:np.asarray(a,float),(log_structure,log_background,entry_structure,entry_background,radius))
    if not (rs.shape==rb.shape==as_.shape==ab.shape==r.shape):raise ValueError('Aligned object vectors required')
    if not all(np.isfinite(x).all() for x in (rs,rb,as_,ab,r)) or (as_<0).any() or (as_>1).any() or (ab<0).any() or (ab>1).any() or (r<0).any():
        raise ValueError('Finite likelihood and physical entry probabilities required')
    logit=np.asarray(zeta)[...,None]-2*np.log1p((r/.1)**2)
    logq=-np.logaddexp(0,-logit);log1q=-np.logaddexp(0,logit)
    numerator=np.logaddexp(logq+rs,log1q+rb)
    with np.errstate(divide='ignore'):
        denominator=np.logaddexp(logq+np.log(as_),log1q+np.log(ab))
    if not np.isfinite(denominator).all():raise ValueError('Zero selection normalization')
    return (numerator-denominator).sum(axis=-1),np.exp(logq+rs-numerator)


def shared_member_counts(probability,weights):
    """Total covariance includes shared global-state uncertainty and Bernoulli noise."""
    p,w=np.asarray(probability,float),np.asarray(weights,float)
    if p.ndim!=2 or w.shape!=(len(p),) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or (w<0).any() or not np.isclose(w.sum(),1):
        raise ValueError('Normalized global weights and state/object probabilities required')
    mean=p.sum(axis=1);expected=float(w@mean)
    conditional=float(w@np.sum(p*(1-p),axis=1))
    shared=float(w@((mean-expected)**2))
    covariance=(conditional+shared)*np.array([[1.,-1.],[-1.,1.]])
    return dict(mean=np.array([expected,p.shape[1]-expected]),covariance=covariance,
        conditional_variance=conditional,shared_variance=shared,per_object_mean=w@p)

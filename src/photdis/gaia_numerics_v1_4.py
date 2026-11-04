"""Convergence-controlled scores for the unchanged histogram distance PDFs."""
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from .baseline_parallax import parallax_check
from .baseline_v1_2 import full_parallax_log_density


def scalar_log_density(pdf,edges,g,y,sigma):
    """Adaptive within-bin integration with log scaling and Gaussian knots."""
    k=np.log(10)/5
    def par(m):return np.exp(k*(m+10-g))
    def log_kernel(m):return -.5*((y-par(m))/sigma)**2-np.log(np.sqrt(2*np.pi)*sigma)
    mode=g-10+np.log(y)/k if y>0 else -np.inf
    active=np.flatnonzero(pdf>0)
    peak=np.array([log_kernel(np.clip(mode,edges[j],edges[j+1])) for j in active])
    logweight=np.log(pdf[active]/np.diff(edges)[active])
    upper=peak+logweight+np.log(np.diff(edges)[active])
    terms=[]
    for j,maximum,lw,ub in zip(active,peak,logweight,upper):
        if ub<np.max(upper)-45:continue
        a,b=edges[j:j+2]
        knots=[a,b]
        for z in [-10,-4,-1,0,1,4,10]:
            p=y+z*sigma
            if p>0:
                m=g-10+np.log(p)/k
                if a<m<b:knots.append(m)
        # Resolve a narrow endpoint tail as well as an interior likelihood peak.
        mm=np.clip(mode,a,b);pp=par(mm)
        slope=abs((y-pp)*pp*k/sigma**2)
        if slope>0:
            for step in [1,4,16,40]:
                m=mm+(1 if mm==a else -1)*step/slope
                if a<m<b:knots.append(m)
        knots=sorted(set(knots));value=0.
        for left,right in zip(knots[:-1],knots[1:]):
            value+=quad(lambda m:np.exp(log_kernel(m)-maximum),left,right,epsabs=1e-12,epsrel=1e-9,limit=100)[0]
        if value>0:terms.append(lw+maximum+np.log(value))
    return float(logsumexp(terms))


def scalar_cdf(pdf,edges,g,y,sigma):
    """Integrate the analytic parallax CDF against Gaussian measurement noise."""
    k=np.log(10)/5
    par=np.exp(k*(edges+10-g));cdf=np.r_[0,np.cumsum(pdf)]
    knots=np.clip((y-par)/sigma,-10,10)
    knots=np.unique(np.r_[-10,knots,10])
    def integrand(z):
        p=y-sigma*z
        f=0. if p<=0 else np.interp(g-10+np.log(p)/k,edges,cdf,left=0,right=1)
        return f*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return float(sum(quad(integrand,a,b,epsabs=1e-11,epsrel=1e-9,limit=100)[0] for a,b in zip(knots[:-1],knots[1:])))


def refined_gaia(pdf,edges,g,y,error):
    """16/32-order comparison, escalating unresolved values independently.

    Precision targets are absolute1e-5 in log density and CDF; this changes no
    trained model, label, object selection, Monte Carlo draw or distance PDF.
    """
    pdf=np.asarray(pdf,float);pdf=pdf/pdf.sum(axis=1)[:,None]
    n=len(g);lo=np.full(n,-np.inf);hi=np.full(n,np.inf)
    a=full_parallax_log_density(pdf,edges,g,y,error,order=16)
    b=full_parallax_log_density(pdf,edges,g,y,error,order=32)
    delta=abs(a-b);initial=delta.copy();orders=np.full(n,32,'i2')
    for order in [64,128]:
        ix=np.flatnonzero(delta>1e-5)
        if not len(ix):break
        value=full_parallax_log_density(pdf[ix],edges,g[ix],y[ix],error[ix],order=order)
        delta[ix]=abs(value-b[ix]);b[ix]=value;orders[ix]=order
    ix=np.flatnonzero(delta>1e-5)
    for j in ix:
        b[j]=scalar_log_density(pdf[j],edges,g[j],y[j],error[j]);orders[j]=0
    p16=parallax_check(pdf,edges,g,y,error,lo,hi,order=16)['pit']
    p32=parallax_check(pdf,edges,g,y,error,lo,hi,order=32)['pit']
    pit_delta=abs(p16-p32);adaptive=pit_delta>1e-5
    for j in np.flatnonzero(adaptive):p32[j]=scalar_cdf(pdf[j],edges,g[j],y[j],error[j])
    assert np.isfinite(b).all() and np.isfinite(p32).all()
    assert np.all((p32>=0)&(p32<=1+1e-10))
    return dict(full_log_density=b,full_pit=np.clip(p32,0,1),log_order=orders,
                log_initial_difference=initial,pit_initial_difference=pit_delta,pit_adaptive=adaptive)

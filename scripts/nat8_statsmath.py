"""Normalized numerical energy scores with exact full-tail density queries."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp,ndtr


def hist_curves(means,weights,sigma,step=.0025,query=None):
    """Linear mean deposition, then the fixed Gaussian kernel; no PDF fit.

    Input can be B conditional forecasts or one flattened parameter mixture.
    Full component parameters remain the actual distribution. This numerical
    CDF/CRPS approximation has a separate half-step/direct-mixture check.
    """
    means=np.atleast_2d(means);weights=np.asarray(weights)
    if weights.ndim==1:weights=np.broadcast_to(weights,means.shape)
    low=float(means.min()-10*sigma);high=float(means.max()+10*sigma)
    if query is not None:low=min(low,float(np.min(query))-.01);high=max(high,float(np.max(query))+.01)
    x=np.arange(np.floor(low/step)*step,np.ceil(high/step)*step+step/2,step);position=(means-x[0])/step;j=np.floor(position).astype(int);f=position-j;offset=np.arange(len(means))[:,None]*len(x)
    hist=np.bincount((offset+j).ravel(),(weights*(1-f)).ravel(),minlength=len(means)*len(x))+np.bincount((offset+j+1).ravel(),(weights*f).ravel(),minlength=len(means)*len(x));hist=hist.reshape(len(means),len(x));density=gaussian_filter1d(hist,sigma/step,axis=1,mode='constant',truncate=10)/step
    cdf=cumulative_trapezoid(density,x,axis=1,initial=0);norm=cdf[:,-1];assert np.max(abs(norm-1))<1e-10
    left=cumulative_trapezoid(cdf*cdf,x,axis=1,initial=0);right=cumulative_trapezoid((1-cdf)**2,x,axis=1,initial=0);crps=left+right[:,-1,None]-right
    return x,cdf,crps,density


def direct_queries(means,lw,sigma,x,a,y):
    """All finite log weights retained even when ordinary exp underflows."""
    a=np.atleast_2d(a);y=np.asarray(y)
    if y.ndim==1:y=y[:,None]
    out=np.empty(y.shape);prob=np.empty(y.shape)
    for b in range(len(a)):
        mm=means-a[b,0]-a[b,1]*x;z=(y[b,:,None]-mm)/sigma;lp=lw-.5*z*z-np.log(sigma*np.sqrt(2*np.pi));out[b]=logsumexp(lp,axis=1);prob[b]=(np.exp(lw)*ndtr(z)).sum(axis=1)
    return prob,out


def quantiles_from_curve(x,cdf):
    p=np.array([.025,.16,.5,.84,.975]);return np.array([np.interp(p,row,x) for row in np.atleast_2d(cdf)])


def curve_values(x,values,y):
    y=np.atleast_2d(y);values=np.atleast_2d(values);assert len(values) in (1,len(y));return np.array([np.interp(row,x,values[0 if len(values)==1 else i]) for i,row in enumerate(y)])


def sparse_CDF_components(means,lw):
    w=np.exp(lw);use=w>=np.finfo(float).eps/len(w);assert 1-w[use].sum()<1e-12
    return means[...,use],w[use]

"""Full normalized Gaussian-mixture likelihood and physical mass measures."""
import numpy as np
from scipy.special import logsumexp,ndtr


def imf_integral(lo,hi):
    lo=np.asarray(lo);hi=np.asarray(hi);assert (hi>lo).all() and (lo>0).all()
    def power(a,b,alpha):return (b**(1-alpha)-a**(1-alpha))/(1-alpha)
    split=.5;left=np.minimum(hi,split);right=np.maximum(lo,split)
    return np.where(left>lo,power(lo,np.maximum(left,lo),1.3),0)+np.where(hi>right,.5*power(right,np.maximum(hi,right),2.3),0)


def mixture(m,e,nodes,prior):
    s=np.maximum(e,.001);iv=1/s**2;total=iv.sum(axis=1)
    means=((m[:,None,:]-nodes[None])*iv[:,None]).sum(axis=2)/total[:,None]
    obs=np.column_stack((m[:,0]-m[:,1],m[:,1]-m[:,2]));cols=np.column_stack((nodes[:,0]-nodes[:,1],nodes[:,1]-nodes[:,2]));diff=obs[:,None]-cols[None]
    v1=s[:,0]**2+s[:,1]**2;v2=s[:,1]**2+s[:,2]**2;cross=-s[:,1]**2;det=v1*v2-cross**2
    chi=(diff[:,:,0]**2*v2[:,None]+diff[:,:,1]**2*v1[:,None]-2*cross[:,None]*diff[:,:,0]*diff[:,:,1])/det[:,None]
    lw=-.5*chi+np.log(prior);lw-=logsumexp(lw,axis=1)[:,None];sigma=np.sqrt(1/total+.1**2)
    return means,lw,sigma,chi.min(axis=1)


def evaluate_mix(y,means,logw,sigma):
    z=(np.asarray(y)[...,None]-means)/np.asarray(sigma)[...,None]
    return np.sum(np.exp(logw)*ndtr(z),axis=-1),logsumexp(logw-.5*z*z-np.log(np.asarray(sigma)[...,None]*np.sqrt(2*np.pi)),axis=-1)


def quantiles(means,logw,sigma):
    out=[]
    for q in (.025,.16,.5,.84,.975):
        lo=means.min(axis=1)-10*sigma;hi=means.max(axis=1)+10*sigma
        for _ in range(45):
            mid=(lo+hi)/2;cdf,_=evaluate_mix(mid,means,logw,sigma);left=cdf<q;lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
        out.append((lo+hi)/2)
    return np.array(out).T


def uniform_curve(x,edges,mass,sigma):
    # Exact convolution of a piecewise-uniform QRF with the reference kernel.
    from scipy.special import log_ndtr
    use=mass>0;lo=(edges[:-1][use,None]-x[None])/sigma;hi=(edges[1:][use,None]-x[None])/sigma;w=np.diff(edges)[use,None]
    flip=lo+hi>0;a=log_ndtr(np.where(flip,-lo,hi));b=log_ndtr(np.where(flip,-hi,lo))
    with np.errstate(divide='ignore',invalid='ignore'):ld=a+np.log(-np.expm1(b-a))
    logpdf=logsumexp(np.log(mass[use,None])+ld-np.log(w),axis=0)
    def G(z):return z*ndtr(z)+np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    cdf=np.sum(mass[use,None]*sigma*(G(-lo)-G(-hi))/w,axis=0)
    return np.clip(cdf,0,1),logpdf


def curve_crps(x,cdf):
    from scipy.integrate import cumulative_trapezoid
    left=cumulative_trapezoid(cdf*cdf,x,initial=0);right=cumulative_trapezoid((1-cdf)**2,x,initial=0)
    return left+right[-1]-right


def row_interp(x,y,values):
    # One grid per experiment, arbitrary query shape beginning with source row.
    k=(y-x[0])/(x[1]-x[0]);assert k.min()>=0 and k.max()<len(x)-1,(k.min(),k.max(),len(x))
    j=np.floor(k).astype(int);f=k-j;ii=np.arange(len(values)).reshape((-1,)+(1,)*(y.ndim-1))
    return values[ii,j]*(1-f)+values[ii,j+1]*f

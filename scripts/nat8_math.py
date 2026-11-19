"""Exact mixture fit with one shared-distance reference, two parameters only."""
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.optimize import minimize


def objective(a,means,lw,sigma,x,mu,counts):
    r=mu+a[0]+a[1]*x[None]-means;iv=1/sigma[:,None]**2;lp=lw-.5*r*r*iv-np.log(sigma[:,None]*np.sqrt(2*np.pi));norm=logsumexp(lp,axis=1);weight=np.exp(lp-norm[:,None]);v=r*iv;g0=np.sum(weight*v,axis=1);g1=np.sum(weight*v*x,axis=1)
    return -float(counts@norm),np.array([counts@g0,counts@g1])


def fit(means,lw,sigma,x,mu,counts,starts,maxiter=500):
    use=counts>0;means=means[use];lw=lw[use];sigma=sigma[use];counts=counts[use];solutions=[]
    for start in starts:
        result=minimize(objective,np.array(start,dtype=float),args=(means,lw,sigma,x,mu,counts),jac=True,method='L-BFGS-B',options=dict(maxiter=maxiter,ftol=1e-12,gtol=1e-6,maxls=40))
        solutions.append(dict(a=result.x,objective=float(result.fun),success=bool(result.success),gradient=float(np.max(abs(result.jac))),iterations=int(result.nit),message=str(result.message)))
    chosen=min(solutions,key=lambda r:r['objective']);assert np.isfinite(chosen['objective']) and np.isfinite(chosen['a']).all()
    a=chosen['a'];r=mu+a[0]+a[1]*x[None]-means;iv=1/sigma[:,None]**2;lp=lw-.5*r*r*iv;w=np.exp(lp-logsumexp(lp,axis=1)[:,None]);v=r*iv;h=np.zeros((2,2))
    for i,j in ((0,0),(0,1),(1,1)):
        xi=1 if i==0 else x;xj=1 if j==0 else x;h[i,j]=np.sum(counts*((w*iv*xi*xj).sum(axis=1)-(w*v*v*xi*xj).sum(axis=1)+(w*v*xi).sum(axis=1)*(w*v*xj).sum(axis=1)));h[j,i]=h[i,j]
    chosen.update(hessian=h,covariance=np.linalg.inv(h) if np.linalg.eigvalsh(h).min()>0 else np.full((2,2),np.nan),solutions=solutions)
    return chosen


def fast_quantiles(means,lw,sigma):
    """All quantiles with at most machine epsilon omitted CDF mass."""
    w=np.exp(lw);use=w>=np.finfo(float).eps/len(w);m=means[...,use];w=w[use];p=np.array([.025,.16,.5,.84,.975]);shape=m.shape[:-1];lo=np.broadcast_to(m.min(axis=-1)[...,None]-10*sigma,shape+(5,)).copy();hi=np.broadcast_to(m.max(axis=-1)[...,None]+10*sigma,shape+(5,)).copy()
    for _ in range(34):
        mid=(lo+hi)/2;cdf=(ndtr((mid[...,None]-m[...,None,:])/sigma)*w).sum(axis=-1);left=cdf<p;lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
    return (lo+hi)/2

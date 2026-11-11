"""Bridge quadrature with the conditional-independence structure of sky fields.

Proposal covariance removes noisy residual correlations BETWEEN fields after
conditioning on global coordinates. This changes quadrature only, never the
posterior or physical model. Exact proposal density and overlap checks remain.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal,multivariate_t


def structured_covariance(samples,global_dimension,block_size):
    x=np.asarray(samples,float);d=x.shape[1];g=int(global_dimension)
    if not 0<g<d or (d-g)%block_size:raise ValueError('Exact global/field coordinate partition required')
    mean=x.mean(axis=0);centered=x-mean;global_x=centered[:,:g]
    cg=np.cov(global_x.T);cg=.98*cg+.02*np.diag(np.diag(cg))
    transform=np.eye(d);base=np.zeros((d,d));base[:g,:g]=cg
    for first in range(g,d,block_size):
        last=first+block_size;local=centered[:,first:last]
        cross=global_x.T@local/(len(x)-1)
        regression=np.linalg.solve(cg,cross)
        residual=local-global_x@regression
        cr=np.cov(residual.T);cr=.98*cr+.02*np.diag(np.diag(cr))
        base[first:last,first:last]=cr;transform[first:last,:g]=regression.T
    covariance=transform@base@transform.T
    if np.linalg.eigvalsh(covariance).min()<=0:raise ValueError('Degenerate structured proposal')
    return mean,covariance


def structured_bridge(logdensity,posterior,global_dimension,block_size,*,proposal_draws=8192,seed=313001):
    x=np.asarray(posterior,float);fitting=x[::2];samples=x[1::2]
    mean,cov=structured_covariance(fitting,global_dimension,block_size)
    gaussian=multivariate_normal(mean,cov);tail=multivariate_t(mean,2*cov,df=7)
    rng=np.random.default_rng(seed);y=gaussian.rvs(size=proposal_draws,random_state=rng)
    heavy=rng.random(proposal_draws)<.05
    if heavy.any():y[heavy]=tail.rvs(size=heavy.sum(),random_state=rng)
    def logproposal(points):return np.logaddexp(np.log(.95)+gaussian.logpdf(points),np.log(.05)+tail.logpdf(points))
    def values(points):
        result=[]
        for z in points:
            try:result.append(logdensity(z))
            except (ValueError,FloatingPointError,OverflowError):result.append(-np.inf)
        return np.asarray(result)
    rp=values(samples)-logproposal(samples);rq=values(y)-logproposal(y)
    if not np.isfinite(rp).all():raise ValueError('Invalid posterior samples')
    def solve(a,b):
        sp=len(a)/(len(a)+len(b));sq=1-sp;z=float(np.median(a));converged=False
        for iteration in range(10000):
            numerator=logsumexp(b-np.logaddexp(np.log(sp)+b,np.log(sq)+z))-np.log(len(b))
            denominator=logsumexp(-np.logaddexp(np.log(sp)+a,np.log(sq)+z))-np.log(len(a))
            update=numerator-denominator
            if abs(update-z)<1e-8:z=update;converged=True;break
            # Damping stabilizes the fixed point without changing its solution.
            z=.5*z+.5*update
        overlap=np.exp(b-np.logaddexp(np.log(sp)+b,np.log(sq)+z))
        return dict(log_integral=float(z),fixed_point_converged=converged,iterations=iteration+1,
            proposal_bridge_ess=float(overlap.sum()**2/(overlap@overlap)))
    result=solve(rp,rq);halves=[solve(rp[j::2],rq[j::2])['log_integral'] for j in (0,1)]
    result.update(half_log_integral_difference=abs(halves[0]-halves[1]),posterior_samples=len(samples),
        proposal_draws=proposal_draws,proposal='0.95 structured Gaussian + 0.05 broad t7',
        log_ratio_posterior=rp,log_ratio_proposal=rq)
    return result

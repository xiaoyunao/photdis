"""Exact-density importance integration; posterior samples train proposal only."""
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.stats import multivariate_normal,multivariate_t
from .structured_evidence import structured_covariance
from .batch_hierarchy import objective_batch


def integrate_population(hierarchy,posterior,draws=32768,seed=315001):
    mean,cov=structured_covariance(np.asarray(posterior),hierarchy.k+2,hierarchy.k)
    normal=multivariate_normal(mean,cov);heavy=multivariate_t(mean,2*cov,df=7)
    rng=np.random.default_rng(seed);points=normal.rvs(size=draws,random_state=rng)
    use=rng.random(draws)<.05
    if use.any():points[use]=heavy.rvs(size=use.sum(),random_state=rng)
    logq=np.logaddexp(np.log(.95)+normal.logpdf(points),np.log(.05)+heavy.logpdf(points))
    logw=-objective_batch(hierarchy,points)-.5*hierarchy.dimension*np.log(2*np.pi)-logq
    weights=softmax(logw);ess=float(1/(weights@weights))
    halves=[float(logsumexp(logw[j::2])-np.log(len(logw[j::2]))) for j in (0,1)]
    return dict(samples=points,weights=weights,log_weights=logw,draw_ids=np.arange(draws,dtype='i8'),
        effective_sample_size=ess,log_integral=float(logsumexp(logw)-np.log(draws)),
        half_log_integral_difference=abs(halves[0]-halves[1]),maximum_weight=float(weights.max()))

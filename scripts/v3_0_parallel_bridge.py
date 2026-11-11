"""Exact parallel evaluation cache for the unchanged structured bridge algorithm."""
import concurrent.futures
import multiprocessing as mp
import numpy as np
from scipy.stats import multivariate_normal,multivariate_t
from photdis.v3.structured_evidence import structured_covariance,structured_bridge
_LOGDENSITY=None


def _value(z):
    try:return _LOGDENSITY(z)
    except (ValueError,FloatingPointError,OverflowError):return -np.inf


def parallel_bridge(logdensity,posterior,global_dimension,block_size,*,proposal_draws=8192,seed=313001,workers=32):
    global _LOGDENSITY
    x=np.asarray(posterior,float);mean,cov=structured_covariance(x[::2],global_dimension,block_size)
    gaussian=multivariate_normal(mean,cov);tail=multivariate_t(mean,2*cov,df=7)
    rng=np.random.default_rng(seed);y=gaussian.rvs(size=proposal_draws,random_state=rng)
    heavy=rng.random(proposal_draws)<.05
    if heavy.any():y[heavy]=tail.rvs(size=heavy.sum(),random_state=rng)
    points=np.concatenate([x[1::2],y]);_LOGDENSITY=logdensity
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        values=list(pool.map(_value,points,chunksize=32))
    cache={z.tobytes():v for z,v in zip(points,values)}
    # Re-run the frozen algorithm, including its deterministic proposal design,
    # exact density, fixed point and half diagnostics. A cache miss fails loudly.
    result=structured_bridge(lambda z:cache[z.tobytes()],x,global_dimension,block_size,
        proposal_draws=proposal_draws,seed=seed)
    result['evaluation_parallel_workers']=workers
    result['cached_density_points']=len(cache)
    return result

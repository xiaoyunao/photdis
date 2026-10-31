"""Adaptive shared-brightness integral, independent of numerator sampling."""
import numpy as np
from scipy.integrate import quad_vec
from photdis.cluster_rgb_shared_distance import log_normal_interval


def adaptive_brightness(mu, g, sg, shared_sigma, tolerance=1e-11):
    """Integrate a common standard-normal shift; factor out (1/2.5)**N.

    The remaining integrand is bounded by the normal density. Truncating at
    +/-12 therefore loses at most 3.6e-33 in absolute probability. This avoids
    underflow from the constant luminosity-density product.
    """
    mu=np.asarray(mu);g=np.asarray(g);sg=np.asarray(sg)
    def integrand(t):
        m=g[:,None]+t*shared_sigma-mu[None,:]
        ll=log_normal_interval((.5-m)/sg[:,None],(3.-m)/sg[:,None])
        return np.exp(ll.sum(axis=0)-.5*t*t)/np.sqrt(2*np.pi)
    probability,error=quad_vec(integrand,-12.,12.,epsabs=tolerance,
        epsrel=tolerance,norm='max',limit=1000)
    if np.any(probability<=0):raise ValueError('Brightness probability underflow')
    return np.log(probability)-len(g)*np.log(2.5),float(error/np.min(probability))

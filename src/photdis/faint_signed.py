"""Selected signed-parallax predictions from native piecewise-uniform PDF bins."""
from functools import lru_cache

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr

K = np.log(10.0) / 5.0
LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


@lru_cache(maxsize=4)
def legendre(order):
    return np.polynomial.legendre.leggauss(order)


def components(edges, mass, order):
    node, weight = legendre(order)
    lo = edges[:, :-1, None]
    width = np.diff(edges)[:, :, None]
    mu = lo + width * (node + 1) / 2
    parallax = np.exp(K * (10 - mu))
    return parallax, mass[:, :, None] * (weight / 2)


def selected_moments(p, w, sigma, limit, shift=0., offset=0.):
    s = sigma[:, None, None]
    upper = limit[:, None, None]
    loc = np.exp(-K * shift) * p + offset
    z = (upper - loc) / s
    accept = ndtr(z)
    correction = s * np.exp(-0.5 * z**2 - LOG_SQRT_2PI)
    prob = (w * accept).sum(axis=(1,2))
    mean = (w * (loc * accept - correction)).sum(axis=(1,2)) / prob
    second = (w * ((loc**2 + s**2) * accept - (loc + upper) * correction)).sum(axis=(1,2)) / prob
    return mean, second-mean**2, prob


def selected_score(p, w, y, sigma, limit, shift=0., offset=0.):
    """Scores and CDF values conditional on the observed weak-parallax cut."""
    loc = np.exp(-K * shift) * p + offset
    s = sigma[:, None, None]
    k = (y[:, None, None] - loc) / s
    with np.errstate(divide='ignore'):
        logw = np.log(w)
    log_accept = logsumexp(logw + log_ndtr((limit[:,None,None] - loc) / s),axis=(1,2))
    log_density = logsumexp(logw - 0.5*k**2,axis=(1,2)) - LOG_SQRT_2PI - np.log(sigma) - log_accept
    pit = np.exp(logsumexp(logw + log_ndtr(k),axis=(1,2)) - log_accept)
    negative = np.exp(logsumexp(logw + log_ndtr(-loc/s),axis=(1,2)) - log_accept)
    return log_density, pit, negative


def scale_weights(slope, offset_slope, variance, block, field_rms, block_rms):
    """Apply inverse working covariance with shared field and block terms."""
    invvar = 1. / variance
    h = offset_slope
    scaled = h * invvar
    normalizer = 1. + block_rms**2 * np.bincount(block, weights=h * scaled)

    def block_inverse(x):
        total = np.bincount(block, weights=scaled * x)
        return invvar*x - scaled * block_rms**2 * total[block] / normalizer[block]

    v_s = block_inverse(slope)
    v_h = block_inverse(h)
    v_s -= v_h*(field_rms**2*np.dot(h,v_s))/(1+field_rms**2*np.dot(h,v_h))
    return v_s / np.dot(slope, v_s)


def selected_draw(edges, cdf, error, limit, shift, offset, rng):
    """Draw directly from the native histograms and reject the same Gaia cut."""
    n, draws = offset.shape
    ix, draw = np.indices((n, draws))
    ix, draw = ix.ravel(), draw.ravel()
    out = np.empty((n, draws))
    for _ in range(256):
        if not len(ix):
            return out
        u = rng.random(len(ix))
        lo = np.zeros(len(ix), dtype=np.int32)
        hi = np.full(len(ix), cdf.shape[1]-1, dtype=np.int32)
        while np.any(lo < hi):
            mid = (lo+hi)//2
            left = cdf[ix,mid]>=u
            hi = np.where(left,mid,hi)
            lo = np.where(left,lo,mid+1)
        mu = edges[ix,lo] + rng.random(len(ix))*(edges[ix,lo+1]-edges[ix,lo])
        y = np.exp(K*(10-mu-shift)) + offset[ix,draw] + error[ix]*rng.standard_normal(len(ix))
        accepted = y < limit[ix]
        out[ix[accepted],draw[accepted]] = y[accepted]
        ix, draw = ix[~accepted],draw[~accepted]
    raise ArithmeticError(f'Weak-selection draws remained unresolved: {len(ix)}')


def simulation(edges, mass, error, limit, block, weight, mean, shift, n,
               field_rms, block_rms, rng):
    reference_cdf = np.cumsum(mass, axis=1)
    reference_cdf[:, -1] = 1.0
    output = []
    for start in range(0, n, 24):
        width = min(n-start,24)
        offset = field_rms*rng.standard_normal(width)
        offset = offset + block_rms*rng.standard_normal((int(block.max())+1,width))[block]
        sample = selected_draw(edges, reference_cdf, error, limit, shift, offset, rng)
        output.extend(weight @ (sample - mean[:,None]))
    return np.asarray(output)

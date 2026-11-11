"""Common continuous-PDF scoring and explicit conditional scientific summaries."""
import numpy as np
from scipy.special import logsumexp, ndtr, log_ndtr
from scipy.stats import norm


def observed_parallax(mu, log_density, parallax, error, order=16):
    """Signed Gaia score from the full log-density, not compressed-CDF tails.

    CDF-error-bounded compression alone cannot certify log scores for conflicting
    observations: tiny posterior tails can dominate such a score. Production
    validation therefore retains the uncompressed marginal log-density grid.
    """
    if not np.isfinite(parallax) or not np.isfinite(error) or error <= 0:
        raise ValueError('Finite signed parallax and positive uncertainty required')
    extra = []
    if parallax > 5*error:
        center = 10-5*np.log10(parallax)
        width = 5/np.log(10)*error/parallax
        extra = center+width*np.linspace(-12, 12, 49)
    mu = np.asarray(mu); log_density = np.asarray(log_density)
    extra = np.asarray(extra)
    edges = np.unique(np.r_[mu, extra[(extra > mu[0]) & (extra < mu[-1])]])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    grid = ((edges[1:]+edges[:-1])/2)[:, None]+np.diff(edges)[:, None]*nodes/2
    integration = np.diff(edges)[:, None]*weights/2
    lp = np.interp(grid.ravel(), mu, log_density)
    logmass = lp+np.log(integration.ravel()); logmass -= logsumexp(logmass)
    predicted = 10**((10-grid.ravel())/5)
    residual = (parallax-predicted)/error
    logscore = logsumexp(logmass+norm.logpdf(parallax, predicted, error))
    pit = np.exp(logmass)@ndtr(residual)
    return dict(log_score=float(logscore), pit=float(pit), negative=bool(parallax < 0))


def normal_reference_score(mu, log_density, reference_mu, error, *, broad_fraction=0., broad_error=2.):
    """Analytic reference-error convolution of the full piecewise log-density.

    Integrating an exponential-linear segment times a Gaussian reduces to a
    shifted Gaussian CDF difference. Tiny probability tails are preserved in log
    space; report CDF compression is deliberately not used for proper scores.
    """
    x = np.asarray(mu); lp = np.asarray(log_density)
    if error <= 0 or not np.isfinite(error): raise ValueError('Positive reference uncertainty required')
    if not 0 <= broad_fraction < 1: raise ValueError('Invalid broad fraction')
    dx = np.diff(x); delta = np.diff(lp); slope = delta/dx
    difference = abs(delta); correction = np.ones_like(difference)
    use = difference > 1e-8
    correction[use] = -np.expm1(-difference[use])/difference[use]
    norm_log = logsumexp(np.maximum(lp[:-1], lp[1:])+np.log(dx)+np.log(correction))
    values = []
    for fraction, sigma in [(1-broad_fraction, error), (broad_fraction, broad_error)]:
        if fraction == 0: continue
        shifted = reference_mu+slope*sigma**2
        lower = (x[:-1]-shifted)/sigma; upper = (x[1:]-shifted)/sigma
        a = np.where(lower > 0, log_ndtr(-lower), log_ndtr(upper))
        b = np.where(lower > 0, log_ndtr(-upper), log_ndtr(lower))
        with np.errstate(divide='ignore', invalid='ignore'):
            logevent = a+np.log(-np.expm1(np.minimum(b-a, 0.)))
        coefficient = lp[:-1]+slope*(reference_mu-x[:-1])+.5*(slope*sigma)**2
        values.append(np.log(fraction)+logsumexp(coefficient+logevent)-norm_log)
    return float(logsumexp(values))


def reference_mu_score(compressed, reference_mu, error, *, broad_fraction=0., broad_error=2.):
    """Exact Gaussian-reference integral under the stored piecewise-CDF density."""
    if error <= 0 or not np.isfinite(error): raise ValueError('Positive reference error required')
    if not 0 <= broad_fraction < 1: raise ValueError('Invalid reference broad fraction')
    score = pit = total = 0.
    def antiderivative(x): return x*ndtr(x)+norm.pdf(x)
    for j, mass in enumerate(compressed['mass']):
        start, stop = compressed['offsets'][j:j+2]
        x = compressed['mu'][start:stop].astype(float)
        cdf = compressed['cdf'][start:stop].astype(float)
        density = mass*np.diff(cdf)/np.diff(x)
        total += mass
        for fraction, sigma in [(1-broad_fraction, error), (broad_fraction, broad_error)]:
            if fraction == 0: continue
            lower = (x[:-1]-reference_mu)/sigma; upper = (x[1:]-reference_mu)/sigma
            # Stable same-tail CDF difference avoids subtracting two values near1.
            event = np.where(lower > 0, ndtr(-lower)-ndtr(-upper), ndtr(upper)-ndtr(lower))
            score += fraction*(density@event)
            primitive = sigma*(antiderivative(-lower)-antiderivative(-upper))
            pit += fraction*(density@primitive)
    return dict(log_score=float(np.log(max(score/total, 1e-300))), pit=float(np.clip(pit/total, 0, 1)))


def distance_summary(mu_quantiles, reference_mu):
    q = np.asarray(mu_quantiles); target = np.asarray(reference_mu)
    valid = np.isfinite(target) & np.isfinite(q).all(axis=1)
    q = q[valid]; target = target[valid]
    if not len(target): return dict(n=0)
    delta = q[:, 2]-target; relative = np.expm1(np.log(10)*delta/5)
    median = np.median(relative)
    return dict(n=len(target), median_relative=float(median),
        nmad_relative=float(1.4826*np.median(abs(relative-median))),
        median_mu_residual=float(np.median(delta)), severe_fraction=float((abs(delta) > 1).mean()),
        coverage68=float(((target >= q[:, 1]) & (target <= q[:, 3])).mean()),
        coverage95=float(((target >= q[:, 0]) & (target <= q[:, 4])).mean()),
        median_width68=float(np.median(q[:, 3]-q[:, 1])),
        median_width95=float(np.median(q[:, 4]-q[:, 0])))


def design_count_covariance(probabilities, strata, stratum_population):
    """SRS-without-replacement Horvitz-Thompson total and design covariance.

    This is the additional sampling-design term, not the conditional latent or
    shared-model count covariance. A stratum with one row cannot estimate its
    variance and is explicitly rejected instead of returning spurious zero.
    """
    p = np.asarray(probabilities); strata = np.asarray(strata)
    total = np.zeros(p.shape[1]); covariance = np.zeros((p.shape[1], p.shape[1]))
    for s in np.unique(strata):
        x = p[strata == s]; n = len(x); population = stratum_population[int(s)]
        if n > population: raise ValueError('Sample exceeds parent stratum')
        total += population*x.mean(axis=0)
        if n == population: continue
        if n < 2: raise ValueError('Design variance unidentifiable from one sampled row')
        residual = x-x.mean(axis=0)
        covariance += population**2*(1-n/population)/n*(residual.T@residual/(n-1))
    return total, covariance

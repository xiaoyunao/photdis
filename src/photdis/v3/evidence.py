"""Overlap-checked bridge evidence for shared response-model marginalization."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_t


def bridge_integral(logdensity, posterior, *, proposal_draws=2048, seed=302901):
    """Integrate a proper-prior unnormalized density using posterior/proposal overlap.

    Callers must include every prior normalization and data-evidence constant.
    Returned half-sample changes diagnose numerical uncertainty, not physical
    model uncertainty. A finite response prior with low ESS remains a limitation.
    """
    x = np.asarray(posterior, float)
    if x.ndim != 2 or len(x) < 100: raise ValueError('Sufficient posterior samples required')
    # Separate proposal fitting from posterior bridge samples to reduce reuse bias.
    fitting = x[::2]; samples = x[1::2]
    mean = fitting.mean(axis=0); covariance = np.cov(fitting.T)
    covariance = np.atleast_2d(covariance)
    eigen, vectors = np.linalg.eigh(covariance)
    if eigen.min() <= 0: raise ValueError('Posterior proposal covariance is singular')
    covariance = (vectors*np.maximum(eigen, eigen.max()*1e-8))@vectors.T
    proposal = multivariate_t(mean, covariance, df=7)
    y = np.asarray(proposal.rvs(size=proposal_draws, random_state=np.random.default_rng(seed))).reshape(proposal_draws, x.shape[1])
    def values(points):
        result = np.empty(len(points))
        for i, point in enumerate(points):
            try: result[i] = logdensity(point)
            except (ValueError, OverflowError, FloatingPointError): result[i] = -np.inf
        return result
    r_p = values(samples)-proposal.logpdf(samples)
    r_q = values(y)-proposal.logpdf(y)
    if not np.isfinite(r_p).all(): raise ValueError('Invalid purported posterior sample')
    def integrate(a, b):
        sp = len(a)/(len(a)+len(b)); sq = 1-sp
        logz = logsumexp(b)-np.log(len(b)); converged = False
        for iteration in range(1000):
            numerator = logsumexp(b-np.logaddexp(np.log(sp)+b, np.log(sq)+logz))-np.log(len(b))
            denominator = logsumexp(-np.logaddexp(np.log(sp)+a, np.log(sq)+logz))-np.log(len(a))
            updated = numerator-denominator
            if abs(updated-logz) < 1e-8:
                logz = updated; converged = True; break
            logz = updated
        overlap = np.exp(b-np.logaddexp(np.log(sp)+b, np.log(sq)+logz))
        ess = float(overlap.sum()**2/(overlap@overlap))
        return dict(log_integral=float(logz), fixed_point_converged=converged,
                    iterations=iteration+1, proposal_bridge_ess=ess)
    result = integrate(r_p, r_q)
    halves = [integrate(r_p[j::2], r_q[j::2])['log_integral'] for j in (0, 1)]
    result.update(half_log_integral_difference=float(abs(halves[0]-halves[1])),
        posterior_samples=len(samples), proposal_draws=proposal_draws,
        log_ratio_posterior=r_p, log_ratio_proposal=r_q)
    return result

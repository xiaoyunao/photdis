"""Shared-nuisance member distance likelihood with exact linear-segment integrals.

Uniform latent Mg support is conditional information inherited from selection.
The no-color control measures how strongly that information constrains distance.
"""
import numpy as np
from scipy.special import log_ndtr, logsumexp
from photdis.cluster_rgb_lower import KNOTS, mean, slope


def log_normal_interval(lo, hi):
    """Stable log(Phi(hi)-Phi(lo)), including far positive/negative tails."""
    lo, hi = np.broadcast_arrays(lo,hi)
    if np.any(hi <= lo):
        raise ValueError('Empty normal interval')
    a = np.where(lo > 0, log_ndtr(-lo), log_ndtr(hi))
    b = np.where(lo > 0, log_ndtr(-hi), log_ndtr(lo))
    return a + np.log(-np.expm1(np.minimum(b-a, -np.finfo(float).eps)))


def member_log_likelihood(model, mu_grid, g0, colors, errors, feh):
    """Return log L(star, mu). Integrate Mg and robust component BEFORE members.

    Observed vector is (g-mu, g-r[, r-z]). Its covariance includes g shared
    with the first color and r shared between colors. Intrinsic scatter lives
    only in the color block. Segment integral is analytic, with no extrapolation.
    """
    nc = colors.shape[1]
    if model['kind'] != 'monotone':
        raise ValueError('Exact integrals require frozen piecewise-linear model')
    transform = np.array([[1.,0,0],[1,-1,0],[0,1,-1]])[:nc+1]
    measurement = np.einsum('ab,nb,cb->nac',transform,errors**2,transform)
    y0 = np.empty((len(g0),len(mu_grid),nc+1))
    y0[:,:,0] = g0[:,None]-mu_grid
    y0[:,:,1:] = colors[:,None,:]
    total = np.full(y0.shape[:2],-np.inf)
    mixture = model.get('contamination',{'weights':[1.], 'scales':[1.]})
    for weight, scale in zip(mixture['weights'],mixture['scales']):
        cov = measurement.copy()
        cov[:,1:,1:] += scale*np.asarray(model['intrinsic'])
        inv = np.linalg.inv(cov)
        _,logdet = np.linalg.slogdet(cov)
        for lo,hi in zip(KNOTS[:-1],KNOTS[1:]):
            gradient = slope(model,np.array([(lo+hi)/2]))[0]
            intercept = mean(model,np.array([lo]),feh)[0]-gradient*lo
            direction = np.r_[1.,gradient]
            y = y0.copy()
            y[:,:,1:] -= intercept
            a = np.einsum('i,nij,j->n',direction,inv,direction)
            b = np.einsum('i,nij,nmj->nm',direction,inv,y)
            mhat = b/a[:,None]
            # Evaluate residual at the optimum to avoid subtraction cancellation.
            residual = y-mhat[:,:,None]*direction
            chi = np.einsum('nmi,nij,nmj->nm',residual,inv,residual)
            logmass = log_normal_interval((lo-mhat)*np.sqrt(a)[:,None],
                                          (hi-mhat)*np.sqrt(a)[:,None])
            ll = (-.5*nc*np.log(2*np.pi)-.5*logdet[:,None]
                  -.5*np.log(a)[:,None]-.5*chi+logmass-np.log(2.5)+np.log(weight))
            total = np.logaddexp(total,ll)
    return total


def prior_member_log_likelihood(mu_grid,g0,sg):
    m = np.asarray(g0)[:,None]-mu_grid
    return log_normal_interval((.5-m)/sg[:,None],(3-m)/sg[:,None])-np.log(2.5)


def marginalize_members(loglike):
    """Input (shared draw, star, mu); return correct and deliberately wrong logs."""
    return (logsumexp(loglike.sum(axis=1),axis=0)-np.log(len(loglike)),
            (logsumexp(loglike,axis=0)-np.log(len(loglike))).sum(axis=0))


def posterior_summary(grid, logprob):
    p = np.exp(logprob-logsumexp(logprob))
    cdf = np.cumsum(p)
    quantiles = np.interp([.025,.16,.5,.84,.975],cdf,grid)
    return dict(q=quantiles.tolist(),width68=float(quantiles[3]-quantiles[1]),
                width95=float(quantiles[4]-quantiles[0]),
                boundary_mass=float(p[:2].sum()+p[-2:].sum())),p

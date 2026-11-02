"""Error-broadened teacher mixtures with jointly propagated Legacy g errors.

These are empirical teacher-scale distributions, not deconvolved population
posteriors or independent calibration of the absolute distance scale.
"""
import numpy as np
from scipy.special import logsumexp
from .giant_kernel import mixture_cdf, mixture_logpdf, mixture_quantiles

COLOR = np.array([[1.,-1.,0.,0.], [0.,1.,-1.,0.], [0.,0.,1.,-1.]])


def predict_mixture(train_bands, train_errors, train_mu, train_mu_error,
                    query_bands, query_errors, dimension=3,
                    bandwidth=.12, magnitude_floor=.15):
    """Condition (C_obs, Z=Mg+epsilon_g), then set mu=g_obs-Z.

    Both train label Mg and query distance use g; Cov(C,Z) includes both
    train and query g variance. All other band errors are independent here.
    An unknown cross-survey teacher covariance is not invented.
    """
    if dimension not in (0,2,3): raise ValueError('Use no-color, grz or grzW1')
    nband = dimension+1 if dimension else 1
    bands = np.asarray(train_bands,float)[:,:nband]
    errors = np.asarray(train_errors,float)[:,:nband]
    qb = np.asarray(query_bands,float)[:nband]
    qe = np.asarray(query_errors,float)[:nband]
    mu = np.asarray(train_mu,float); se = np.asarray(train_mu_error,float)
    if (not len(mu) or any(np.any(~np.isfinite(x)) for x in (bands,errors,qb,qe,mu,se))
            or np.any(errors<0) or np.any(qe<0) or np.any(se<=0)):
        raise ValueError('Finite measurements and positive teacher widths required')
    mg = bands[:,0]-mu
    variance = errors[:,0]**2 + se**2 + magnitude_floor**2 + qe[0]**2
    means = mg.copy(); weights = np.full(len(mu),1/len(mu)); outside = False
    if dimension:
        transform = COLOR[:dimension,:nband]
        colors = bands @ transform.T; query = qb @ transform.T
        covariance = np.einsum('ab,nb,cb->nac',transform,errors**2,transform)
        covariance += np.einsum('ab,b,cb->ac',transform,qe**2,transform)
        covariance += bandwidth**2 * np.eye(dimension)
        # Cov(C_obs, Mg+epsilon_g); shared-g terms must add before conditioning.
        cross = (errors[:,0]**2+qe[0]**2)[:,None]*transform[:,0]
        diff = query-colors
        solved = np.linalg.solve(covariance,cross[...,None])[...,0]
        means += np.einsum('ni,ni->n',solved,diff)
        variance -= np.einsum('ni,ni->n',solved,cross)
        logdet = np.linalg.slogdet(covariance)[1]
        inverse_diff = np.linalg.solve(covariance,diff[...,None])[...,0]
        lw = -.5*(logdet+np.einsum('ni,ni->n',diff,inverse_diff))
        weights = np.exp(lw-logsumexp(lw))
        outside = bool(np.any((query<colors.min(axis=0))|(query>colors.max(axis=0))))
    if np.any(variance<=0): raise ValueError('Nonpositive conditional variance')
    return {'means':qb[0]-means, 'sigmas':np.sqrt(variance), 'weights':weights,
            'effective_n':float(1/np.sum(weights**2)), 'max_weight':float(weights.max()),
            'outside_training_box':outside}


def evaluate_mixture(mixture, reference_mu, reference_sigma):
    means,sigmas,weights = (mixture[k] for k in ('means','sigmas','weights'))
    q = mixture_quantiles(means,sigmas,weights)
    center = float(q[2]); residual = center-reference_mu
    widened = np.hypot(sigmas,reference_sigma)
    return {'quantiles':q.tolist(), 'delta_mu':float(residual),
            'relative_residual':float(10**(residual/5)-1),
            'raw_cover68':bool(q[1]<=reference_mu<=q[3]),
            'raw_cover95':bool(q[0]<=reference_mu<=q[4]),
            'raw_width68':float(q[3]-q[1]), 'raw_width95':float(q[4]-q[0]),
            'log_score':mixture_logpdf(reference_mu,means,sigmas,weights,reference_sigma),
            'convolved_pit':mixture_cdf(reference_mu,means,widened,weights),
            'effective_n':mixture['effective_n'], 'max_weight':mixture['max_weight'],
            'outside_training_box':mixture['outside_training_box']}

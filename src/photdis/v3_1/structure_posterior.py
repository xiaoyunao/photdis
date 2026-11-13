"""Marginalize one shared response, structure distance and spatial-odds parameter."""
import numpy as np
from scipy.special import logsumexp
from .structure import selected_structure_loglike


def common_structure_posterior(log_structure,log_background,entry_structure,entry_background,
                               radius,common_mu,common_logweight,zeta,zeta_logweight):
    """Exact finite quadrature moments; global states are shared by all objects.

    The finite SED/calibration prior is uniform over its entire supplied set.
    Callers must verify that this is the complete registered set, and establish
    distance/odds quadrature convergence; this routine does not invent that check.
    """
    rs,rb,as_,ab,rad,mu,lmu,z,lz=map(lambda a:np.asarray(a,float),
        (log_structure,log_background,entry_structure,entry_background,radius,common_mu,common_logweight,zeta,zeta_logweight))
    if rs.ndim!=3 or as_.shape!=rs.shape or rb.shape!=rs.shape[:2] or ab.shape!=rb.shape or rad.shape!=(rs.shape[1],):
        raise ValueError('Response/object/common-distance arrays required')
    s,n,m=rs.shape
    if mu.shape!=(m,) or lmu.shape!=(m,) or z.ndim!=1 or lz.shape!=z.shape or s<1 or n<1:
        raise ValueError('Aligned nonempty global quadrature required')
    if not np.isclose(logsumexp(lmu),0,atol=1e-12) or not np.isclose(logsumexp(lz),0,atol=1e-12):
        raise ValueError('Normalized shared quadrature priors required')
    likelihood=np.empty((s,m,len(z)))
    for response in range(s):
        for j in range(m):
            likelihood[response,j],_=selected_structure_loglike(rs[response,:,j],rb[response],as_[response,:,j],ab[response],rad,z)
    unnormalized=likelihood+lmu[None,:,None]+lz[None,None]-np.log(s)
    evidence=float(logsumexp(unnormalized));weights=np.exp(unnormalized-evidence)
    per_object=np.zeros(n);conditional_mean=np.empty_like(weights);conditional_variance=np.empty_like(weights)
    for response in range(s):
        for j in range(m):
            _,probability=selected_structure_loglike(rs[response,:,j],rb[response],as_[response,:,j],ab[response],rad,z)
            conditional_mean[response,j]=probability.sum(axis=1)
            conditional_variance[response,j]=np.sum(probability*(1-probability),axis=1)
            per_object+=weights[response,j]@probability
    mean=float(np.sum(weights*conditional_mean));conditional=float(np.sum(weights*conditional_variance))
    shared=float(np.sum(weights*(conditional_mean-mean)**2));covariance=(conditional+shared)*np.array([[1.,-1.],[-1.,1.]])
    common_mass=weights.sum(axis=(0,2));order=np.argsort(mu);cdf=np.cumsum(common_mass[order]);cdf[-1]=1
    quantiles=mu[order[np.searchsorted(cdf,[.025,.16,.5,.84,.975])]]
    return dict(log_evidence=np.array(evidence),global_weights=weights,log_likelihood=likelihood,
        response_weights=weights.sum(axis=(1,2)),common_mu=mu,common_mu_probability=common_mass,common_mu_quantiles=quantiles,
        zeta=z,zeta_probability=weights.sum(axis=(0,1)),per_object_member_probability=per_object,
        mean=np.array([mean,n-mean]),covariance=covariance,conditional_covariance=conditional*np.array([[1.,-1.],[-1.,1.]]),
        shared_covariance=shared*np.array([[1.,-1.],[-1.,1.]]),conditional_member_mean=conditional_mean,
        conditional_member_variance=conditional_variance,n=np.array(n))

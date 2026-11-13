"""Realized membership counts with one posterior global state per catalog."""
import numpy as np
from scipy.special import expit


def realized_structure_counts(log_structure,log_background,radius,zeta,weights,*,draws,seed):
    rs,rb,rad,z,w=map(lambda x:np.asarray(x,float),(log_structure,log_background,radius,zeta,weights))
    if rs.ndim!=3 or rb.shape!=rs.shape[:2] or rad.shape!=(rs.shape[1],) or w.shape!=(rs.shape[0],rs.shape[2],len(z)):
        raise ValueError('Aligned response/object/distance/odds posterior required')
    if not all(np.isfinite(x).all() for x in (rs,rb,rad,z,w)) or (rad<0).any() or (w<0).any() or not np.isclose(w.sum(),1,atol=1e-12,rtol=0):
        raise ValueError('Finite responses and normalized global posterior required')
    if draws<2 or draws%2:raise ValueError('Even number of at least two draws required')
    rng=np.random.default_rng(seed); n=rs.shape[1]
    # Exactly ONE index per realized catalog, shared by all its objects.
    flat=rng.choice(w.size,size=draws,p=w.ravel()); response,distance,odds=np.unravel_index(flat,w.shape)
    counts=np.empty((draws,2),dtype='i8'); spatial=-2*np.log1p((rad/.1)**2)
    for start in range(0,draws,512):
        sl=slice(start,min(start+512,draws)); ri,mi,zi=response[sl],distance[sl],odds[sl]
        probability=expit(z[zi,None]+spatial+rs[ri,:,mi]-rb[ri])
        count=(rng.random(probability.shape)<probability).sum(axis=1)
        counts[sl,0]=count; counts[sl,1]=n-count
    return dict(realized_counts=counts,global_flat_index=flat.astype('i8'),
        response_index=response.astype('i8'),distance_index=distance.astype('i8'),odds_index=odds.astype('i8'))


def count_draw_diagnostics(counts,exact_mean,exact_covariance):
    counts=np.asarray(counts); mean=np.asarray(exact_mean,float); covariance=np.asarray(exact_covariance,float)
    if counts.ndim!=2 or counts.shape[1]!=2 or counts.dtype.kind not in 'iu' or len(counts)<2 or len(counts)%2:
        raise ValueError('Even complete int member/background catalogs required')
    n=int(counts[0].sum())
    if (counts<0).any() or not (counts.sum(axis=1)==n).all():raise ValueError('Each realized catalog must conserve count')
    if mean.shape!=(2,) or covariance.shape!=(2,2):raise ValueError('Two-class exact moments required')
    empirical=counts.mean(axis=0); variance=float(counts[:,0].var(ddof=1))
    se=float(np.sqrt(max(covariance[0,0],0)/len(counts)))
    halves=[counts[j::2,0] for j in (0,1)]
    cdfs=[np.cumsum(np.bincount(x,minlength=n+1))/len(x) for x in halves]
    cdf_delta=float(abs(cdfs[0]-cdfs[1]).max())
    quantiles=np.quantile(counts,[.025,.16,.5,.84,.975],axis=0,method='inverted_cdf').T.astype('i8')
    half_q=[np.quantile(x,[.025,.16,.5,.84,.975],method='inverted_cdf') for x in halves]
    q_delta=int(abs(half_q[0]-half_q[1]).max())
    v_delta=float(abs(halves[0].var(ddof=1)-halves[1].var(ddof=1))/max(covariance[0,0],1))
    mean_delta=float(abs(empirical-mean).max())
    passed=mean_delta<5*se+.05 and cdf_delta<.01 and q_delta<=2 and v_delta<.05
    return dict(draws=len(counts),n=n,exact_mean=mean.tolist(),empirical_mean=empirical.tolist(),
        exact_member_variance=float(covariance[0,0]),empirical_member_variance=variance,
        maximum_mean_error=mean_delta,mean_MC_standard_error=se,CDF_half_delta=cdf_delta,
        quantile_half_count_delta=q_delta,relative_variance_half_delta=v_delta,
        quantiles=quantiles.tolist(),count_conservation=True,passed=bool(passed))

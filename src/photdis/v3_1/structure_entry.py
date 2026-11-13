"""Direct observed-context entry for shared narrow and broad structure priors."""
import numpy as np
from scipy.special import log_ndtr,ndtri_exp,ndtri
from scipy.stats import qmc
from photdis.v3.selection import OPTICAL
from .candidate_selection import candidate_thresholds


def proposal_entry(model,raw_error,transmission,calibration_mag,means,sigmas,*,
                   proposal_mean,proposal_sigma,draws=524288,seed=320211):
    """One proper Gaussian proposal, exact density ratios, no observed-context interpolation.

    All targets must be narrower than the proposal. The proposal cancels in
    each integral. Raw finite-QMC estimates are retained even above one; the
    production physical/numerical acceptance gate must independently reject them.
    """
    if draws<2 or draws&(draws-1):raise ValueError('Power-of-two Sobol size required')
    means,sigmas=np.asarray(means,float),np.asarray(sigmas,float)
    if means.ndim!=1 or means.shape!=sigmas.shape or not np.isfinite(means).all() or not np.isfinite(sigmas).all() or (sigmas<=0).any() or (sigmas>=proposal_sigma).any():
        raise ValueError('Finite narrower Gaussian targets required')
    a=.5/proposal_sigma**2-.5/sigmas**2
    b=(means-proposal_mean)/sigmas**2
    c=np.log(proposal_sigma/sigmas)-.5*((means-proposal_mean)/sigmas)**2
    vertex=-b/(2*a);logmax=c-b*b/(4*a)
    error,_=candidate_thresholds(raw_error,transmission,calibration_mag);lower=5*error
    u=qmc.Sobol(5,scramble=True,seed=seed).random_base2(int(np.log2(draws)))
    u=np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps)
    offset=proposal_sigma*ndtri(u[:,1]);mu=proposal_mean+offset
    amplitude=10**(-.4*(mu-10));logu=np.log(u[:,2:])
    prior=model.weights[:,None]*model.classes;prior/=prior.sum(axis=0)[None]
    intrinsic=model.means[:,OPTICAL];covariance=model.covariances[:,OPTICAL][:,:,OPTICAL]
    alpha=np.zeros((3,len(means)));ess=np.zeros_like(alpha)
    for cls in range(3):
        cumulative=np.cumsum(prior[:,cls]);cumulative[-1]=1.
        index=np.searchsorted(cumulative,u[:,0],side='right');value=np.zeros(draws)
        for fraction,factor in [(1-model.binary_reference_fraction,1.),(model.binary_reference_fraction,model.binary_brightness_factor)]:
            amplitude_binary=amplitude*factor;mean=intrinsic[index]*amplitude_binary[:,None]
            cov=covariance[index]*amplitude_binary[:,None,None]**2+np.diag(error**2)[None]
            chol=np.linalg.cholesky(cov);z=np.zeros((draws,3));logweight=np.zeros(draws)
            for band in range(3):
                center=mean[:,band]+np.einsum('ni,ni->n',chol[:,band,:band],z[:,:band])
                tail=log_ndtr((center-lower[band])/chol[:,band,band]);logweight+=tail
                if band<2:z[:,band]=-ndtri_exp(tail+logu[:,band])
            value+=fraction*np.exp(logweight)
        total=np.zeros(len(means));second=np.zeros_like(total)
        for start in range(0,draws,8192):
            stop=min(draws,start+8192)
            ratio=np.exp(logmax[:,None]+a[:,None]*(offset[None,start:stop]-vertex[:,None])**2)
            total+=ratio@value[start:stop];second+=(ratio*ratio)@(value[start:stop]**2)
        alpha[cls]=total/draws
        ess[cls]=np.divide(total*total,second,out=np.zeros_like(total),where=second>0)
    if not np.isfinite(alpha).all() or (alpha<0).any():raise ValueError('Invalid importance integral')
    return alpha,dict(maximum_radial_density_ratio=np.exp(logmax),integration_weight_ess=ess,
        proposal_mean=np.array(proposal_mean),proposal_sigma=np.array(proposal_sigma))

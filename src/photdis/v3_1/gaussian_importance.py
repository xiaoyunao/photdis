"""Reuse one proper radial proposal for all shared Gaussian correction draws."""
import numpy as np
from scipy.special import log_ndtr,ndtri_exp,ndtri
from scipy.stats import qmc
from photdis.v3.selection import OPTICAL
from .candidate_selection import candidate_thresholds
from .gaussian_entry import transformed_radial_parameters


def gaussian_ratio_coefficients(location,scale,proposal_mean=15.,proposal_sigma=6.):
    location,scale=np.asarray(location,float),np.asarray(scale,float)
    if not np.isfinite(location).all() or not np.isfinite(scale).all() or (scale<=0).any() or (scale>=proposal_sigma).any():
        raise ValueError('Finite narrower Gaussian targets required for a bounded ratio')
    a=.5/proposal_sigma**2-.5/scale**2
    b=location/scale**2-proposal_mean/proposal_sigma**2
    c=-.5*(location/scale)**2+.5*(proposal_mean/proposal_sigma)**2+np.log(proposal_sigma/scale)
    maximum=np.exp(c-b*b/(4*a))
    return a,b,c,maximum


def shared_gaussian_entry(model,raw_error,transmission,calibration_mag,means,sigmas,corrections,*,
                          support=None,candidate=False,draws=524288,seed=319711):
    """Exact importance identity; QMC precision must be established separately.

    The N(15,6) proposal is not a scientific distance prior. Its density cancels
    in each target radial integral. No correction-draw weights are flattened or
    updated here. All binary states and fixed conditional kernel weights remain.
    """
    if draws<2 or draws&(draws-1):raise ValueError('Power-of-two Sobol size required')
    error,lower=candidate_thresholds(raw_error,transmission,calibration_mag)
    if candidate and support is None:raise ValueError('Exact candidate support required')
    if not candidate:lower=5*error
    parameters=[transformed_radial_parameters(means,sigmas,p) for p in corrections]
    locations=np.array([p[0] for p in parameters]);scales=np.array([p[1] for p in parameters])
    ca,cb,cc,maximum=gaussian_ratio_coefficients(locations,scales)
    u=qmc.Sobol(5,scramble=True,seed=seed).random_base2(int(np.log2(draws)));u=np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps)
    mu=15+6*ndtri(u[:,1]);amplitude=10**(-.4*(mu-10));logu=np.log(u[:,2:])
    prior=model.weights[:,None]*model.classes;prior/=prior.sum(axis=0)[None]
    intrinsic=model.means[:,OPTICAL];covariance=model.covariances[:,OPTICAL][:,:,OPTICAL]
    alpha=np.zeros(locations.shape);bounds=np.zeros_like(alpha);ess=np.zeros_like(alpha)
    for c in range(3):
        cumulative=np.cumsum(prior[:,c]);cumulative[-1]=1.
        index=np.searchsorted(cumulative,u[:,0],side='right');value=np.zeros(draws);omitted=np.zeros(draws)
        for fraction,factor in [(1-model.binary_reference_fraction,1.),(model.binary_reference_fraction,model.binary_brightness_factor)]:
            a=amplitude*factor;mean=intrinsic[index]*a[:,None]
            cov=covariance[index]*a[:,None,None]**2+np.diag(error**2)[None]
            chol=np.linalg.cholesky(cov);z=np.zeros((draws,3));logweight=np.zeros(draws)
            for b in range(3):
                center=mean[:,b]+np.einsum('ni,ni->n',chol[:,b,:b],z[:,:b])
                tail=log_ndtr((center-lower[b])/chol[:,b,b]);logweight+=tail
                if b<2 or candidate:z[:,b]=-ndtri_exp(tail+logu[:,b])
            weight=np.exp(logweight)
            if candidate:
                active=weight>1e-14;admitted=np.zeros(draws,bool)
                if active.any():
                    flux=mean[active]+np.einsum('nij,nj->ni',chol[active],z[active])
                    if (flux<=0).any() or not np.isfinite(flux).all():raise RuntimeError('Invalid selected flux')
                    selected=np.asarray(support(22.5-2.5*np.log10(flux)+np.asarray(calibration_mag)))
                    if selected.dtype.kind!='b' or selected.shape!=(active.sum(),):raise ValueError('Boolean support required')
                    admitted[active]=selected
                omitted+=fraction*np.where(active,0.,weight);weight*=admitted
            value+=fraction*weight
        total=np.zeros((len(corrections),len(means)));second=np.zeros_like(total)
        for start in range(0,draws,8192):
            stop=min(draws,start+8192);x=mu[start:stop]
            ratio=np.exp(ca[:,c,:,None]*x[None,None]**2+cb[:,c,:,None]*x[None,None]+cc[:,c,:,None])
            assert (ratio<=maximum[:,c,:,None]*(1+1e-12)).all()
            total+=ratio@value[start:stop];second+=(ratio*ratio)@(value[start:stop]**2)
        alpha[:,c]=total/draws;bounds[:,c]=maximum[:,c]*omitted.mean()
        ess[:,c]=np.divide(total*total,second,out=np.zeros_like(total),where=second>0)
    if not np.isfinite(alpha).all() or (alpha<0).any() or (alpha>1.001).any():raise RuntimeError('Invalid entry integral; retain unresolved QMC')
    return alpha,dict(pruning_bound=bounds,integration_weight_ess=ess,maximum_radial_density_ratio=maximum)

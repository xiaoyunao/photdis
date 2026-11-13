"""Joint Sobol integration of class entry under Gaussian radial bases."""
import numpy as np
from scipy.special import log_ndtr,ndtri_exp,ndtri
from scipy.stats import qmc
from photdis.v3.selection import OPTICAL
from .candidate_selection import candidate_thresholds


def transformed_radial_parameters(means,sigmas,correction):
    """Withdraw the luminosity convolution into the Gaussian integration prior.

    Integral L(mu-delta-t) N(t;0,sigma) N(mu;m,s) dmu dt
    = Integral L(v) N(v;m-delta,sqrt(s*s+sigma*sigma)) dv.
    The identity applies equally to the entry probability, before normalization.
    """
    means,sigmas,correction=map(lambda x:np.asarray(x,float),(means,sigmas,correction))
    if means.ndim!=1 or sigmas.shape!=means.shape or correction.shape!=(3,2): raise ValueError('Gaussian bases and three class corrections required')
    if not all(np.isfinite(x).all() for x in (means,sigmas,correction)) or (sigmas<=0).any(): raise ValueError('Finite positive radial scales required')
    if (abs(correction[:,0])>1).any() or (correction[:,1]<0).any() or (correction[:,1]>1).any(): raise ValueError('Registered correction bounds exceeded')
    return means[None]-correction[:,0,None],np.sqrt(sigmas[None]**2+correction[:,1,None]**2)


def class_gaussian_entry(model,raw_error,transmission,calibration_mag,means,sigmas,*,
                         support=None,candidate=False,correction=None,draws=32768,seed=319701):
    """Joint kernel/radial/truncated-flux QMC; binary states integrated exactly.

    No distance grid or hard-support interpolation. This changes only numerical
    integration; independent seeds/levels and original quadrature comparisons
    remain required. The reported ESS describes integration weights, not data or
    response-posterior support. Local dust must already be in the integration model.
    """
    if draws<2 or draws&(draws-1): raise ValueError('Power-of-two Sobol draws required')
    error,threshold=candidate_thresholds(raw_error,transmission,calibration_mag)
    if not candidate: threshold=5*error
    elif support is None: raise ValueError('Exact hard QRF support required for candidate admission')
    correction=np.zeros((3,2)) if correction is None else correction
    location,width=transformed_radial_parameters(means,sigmas,correction)
    u=qmc.Sobol(5,scramble=True,seed=seed).random_base2(int(np.log2(draws)))
    u=np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps)
    radial_z=ndtri(u[:,1]); logu=np.log(u[:,2:])
    prior=model.weights[:,None]*model.classes; prior/=prior.sum(axis=0)[None]
    intrinsic=model.means[:,OPTICAL]; covariance=model.covariances[:,OPTICAL][:,:,OPTICAL]
    output=np.zeros(location.shape); bound=np.zeros_like(output); ess=np.zeros_like(output)
    for c in range(3):
        cumulative=np.cumsum(prior[:,c]);cumulative[-1]=1.
        kernel=np.searchsorted(cumulative,u[:,0],side='right')
        for j in range(location.shape[1]):
            mu=location[c,j]+width[c,j]*radial_z; amplitude=10**(-.4*(mu-10))
            value=np.zeros(draws); omitted=np.zeros(draws)
            for fraction,factor in [(1-model.binary_reference_fraction,1.),(model.binary_reference_fraction,model.binary_brightness_factor)]:
                a=amplitude*factor; mean=intrinsic[kernel]*a[:,None]
                cov=covariance[kernel]*a[:,None,None]**2+np.diag(error**2)[None]
                chol=np.linalg.cholesky(cov); z=np.zeros((draws,3)); logweight=np.zeros(draws)
                for b in range(3):
                    center=mean[:,b]+np.einsum('ni,ni->n',chol[:,b,:b],z[:,:b])
                    tail=log_ndtr((center-threshold[b])/chol[:,b,b]);logweight+=tail
                    if b<2 or candidate: z[:,b]=-ndtri_exp(tail+logu[:,b])
                weight=np.exp(logweight); active=weight>1e-14
                if candidate:
                    admitted=np.zeros(draws,bool)
                    if active.any():
                        flux=mean[active]+np.einsum('nij,nj->ni',chol[active],z[active])
                        if (flux<=0).any() or not np.isfinite(flux).all(): raise RuntimeError('Invalid selected native flux')
                        mags=22.5-2.5*np.log10(flux)+np.asarray(calibration_mag)
                        selected=np.asarray(support(mags))
                        if selected.dtype.kind!='b' or selected.shape!=(active.sum(),): raise ValueError('Boolean exact support required')
                        admitted[active]=selected
                    omitted+=fraction*np.where(active,0.,weight)
                    weight*=admitted
                value+=fraction*weight
            output[c,j]=value.mean();bound[c,j]=omitted.mean()
            ess[c,j]=0. if not (value>0).any() else value.sum()**2/(value@value)
    if not np.isfinite(output).all() or (output<0).any() or (output>1+1e-12).any(): raise RuntimeError('Invalid class entry integral')
    return output,dict(pruning_bound=bound,integration_weight_ess=ess)

"""Posterior negative latent-flux support, distinct from kernel-prior proxies."""
import numpy as np
from scipy.special import ndtr
from photdis.v3.likelihood import trapezoid_weights


def negative_flux_support(model, observation, posterior, *, bands=None, calibration_mag=None):
    """Gaussian conditional marginal probabilities and union-event bounds.

    Matches the frozen likelihood's mean-linearized dust covariance. The latent
    variable is the binary-scaled, unextincted flux at1kpc; multiplying by positive
    brightness leaves its sign unchanged. Censored measurements need a truncated
    Gaussian integral and are explicitly unsupported by this diagnostic.
    """
    mask=np.array(observation['band_mask'],bool)
    if bands is not None:mask &= np.asarray(bands,bool)
    if not mask.any():raise ValueError('At least one measured band required')
    upper=np.asarray(observation.get('upper_limit',np.zeros(6,bool)))
    if upper[mask].any():raise ValueError('Censored latent-flux conditional requires truncated Gaussian integration')
    k=len(model.weights); grid=posterior.mu
    if posterior.joint_density.shape != (2*k*3,len(grid)):
        raise ValueError('Binary x component x class joint posterior required')
    mass=posterior.joint_density.reshape(2,k,3,-1).sum(axis=2)*trapezoid_weights(grid)
    if not np.isclose(mass.sum(),1):raise ValueError('Normalized joint posterior required')
    tr=np.asarray(observation['transmission'],float).copy()
    if calibration_mag is not None:tr *= 10**(-.4*np.asarray(calibration_mag))
    amplitude=10**(-.4*(grid-10))
    transform=amplitude[:,None]*tr[None,mask]
    flux=np.asarray(observation['flux'])[mask]; error=np.asarray(observation['flux_error'])[mask]
    if not np.isfinite(flux).all() or not np.isfinite(error).all() or (error<=0).any():
        raise ValueError('Valid signed observed flux and positive errors required')
    ebv=float(observation.get('EBV',0.)); dust=float(observation.get('dust_sigma_ebv',.1*max(ebv,0.)))
    if not np.isfinite(dust) or dust<0:raise ValueError('Valid dust uncertainty required')
    if dust and ebv<=0:raise ValueError('Positive EBV needed for measured extinction coefficients')
    coefficients=-2.5*np.log10(np.asarray(observation['transmission'])[mask])/ebv if dust else np.zeros(mask.sum())
    marginal=np.zeros(6); lower=upper_bound=0.
    for b,brightness in enumerate((1.,model.binary_brightness_factor)):
        for component in range(k):
            weight=mass[b,component]
            if weight.sum()==0:continue
            mean=model.means[component,:6]*brightness
            cov=model.covariances[component,:6,:6]*brightness**2
            observed_mean=transform*mean[None,mask]
            variance=cov[np.ix_(mask,mask)][None]*transform[:,:,None]*transform[:,None,:]+np.diag(error**2)[None]
            if dust:
                perturbation=.4*np.log(10)*dust*observed_mean*coefficients
                variance += perturbation[:,:,None]*perturbation[:,None,:]
            cross=cov[:,mask][None]*transform[:,None,:]
            gain=np.linalg.solve(variance,np.swapaxes(cross,1,2))
            conditional_mean=mean[None]+np.einsum('mdf,mf->md',cross,np.linalg.solve(variance,(flux-observed_mean)[...,None])[...,0])
            conditional_variance=np.diag(cov)[None]-np.einsum('mdf,mfd->md',cross,gain)
            # Tiny subtractive roundoff at high S/N is not physical negative variance.
            if (conditional_variance < -1e-10*np.maximum(np.diag(cov),1e-30)[None]).any():
                raise ValueError('Negative Gaussian conditional variance')
            sigma=np.sqrt(np.maximum(conditional_variance,np.finfo(float).tiny))
            probability=ndtr(-conditional_mean/sigma)
            marginal += weight@probability
            lower += weight@probability.max(axis=1)
            upper_bound += weight@np.minimum(probability.sum(axis=1),1)
    return dict(probability_by_band=marginal,any_negative_lower=float(lower),
        any_negative_upper=float(upper_bound),
        scope='Gaussian conditional under the frozen mean-linearized dust approximation; any-band event bounded, not assumed independent')

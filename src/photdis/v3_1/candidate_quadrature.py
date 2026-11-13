"""Bounded support-query pruning for higher-order candidate-entry quadrature."""
import numpy as np
from scipy.special import log_ndtr, ndtri_exp
from scipy.stats import qmc
from photdis.v3.selection import OPTICAL
from photdis.v3_1.candidate_selection import candidate_thresholds


def bounded_selected_probability(mean, covariance, lower, calibration_mag, *, support,
                                 draws=2048, seed=317101, support_weight_floor=1e-14):
    """Same truncated Gaussian QMC, with a certified omitted-weight upper bound.

    Only hard-support queries for individually negligible Gaussian survival
    weights are skipped. Return both the lower estimate and the omitted weight;
    the original unpruned QMC probability lies between their sum endpoints.
    This bounds pruning error, not the independent QMC integration error.
    """
    mean,covariance = np.asarray(mean,float),np.asarray(covariance,float)
    lower,cal = np.asarray(lower,float),np.asarray(calibration_mag,float)
    if mean.shape[-1:] != (3,) or covariance.shape != mean.shape[:-1]+(3,3): raise ValueError('Aligned Gaussian required')
    if lower.shape != (3,) or cal.shape != (3,) or (lower <= 0).any(): raise ValueError('Positive thresholds required')
    if not all(np.isfinite(x).all() for x in (mean,covariance,lower,cal)): raise ValueError('Finite Gaussian required')
    if draws < 2 or draws & (draws-1): raise ValueError('Power-of-two Sobol draws required')
    if not 0 <= support_weight_floor <= 1e-12: raise ValueError('Only negligible support pruning is allowed')
    shape = mean.shape[:-1]; mean = mean.reshape(-1,3); chol = np.linalg.cholesky(covariance.reshape(-1,3,3))
    u = qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(np.log2(draws)))
    logu = np.log(np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps))
    z = np.zeros((len(mean),draws,3)); logweight = np.zeros(z.shape[:2])
    for j in range(3):
        center = mean[:,j,None]+np.einsum('ni,nsi->ns',chol[:,j,:j],z[:,:,:j])
        tail = log_ndtr((center-lower[j])/chol[:,j,j,None])
        z[:,:,j] = -ndtri_exp(tail+logu[None,:,j]); logweight += tail
    weight = np.exp(logweight); active = weight > support_weight_floor
    omitted = np.where(active,0.,weight).mean(axis=1)
    admitted = np.zeros(weight.shape,bool)
    if active.any():
        i,j = np.nonzero(active)
        flux = mean[i]+np.einsum('nij,nj->ni',chol[i],z[i,j])
        if (flux <= 0).any() or not np.isfinite(flux).all(): raise RuntimeError('Invalid truncated selected flux')
        magnitudes = 22.5-2.5*np.log10(flux)+cal
        chosen = np.asarray(support(magnitudes))
        if chosen.dtype.kind != 'b' or chosen.shape != (len(i),): raise ValueError('Boolean support required')
        admitted[i,j] = chosen
    return (weight*admitted).mean(axis=1).reshape(shape),omitted.reshape(shape)


def bounded_component_entry(model, raw_error, transmission, calibration_mag, mu, *, support, draws, seed):
    error,lower = candidate_thresholds(raw_error,transmission,calibration_mag)
    means = model.means[:,OPTICAL]; covariance = model.covariances[:,OPTICAL][:,:,OPTICAL]
    amplitude = 10**(-.4*(np.asarray(mu)-10.)); result = np.empty((2,len(means),len(amplitude))); bound = np.empty_like(result)
    for b,factor in enumerate((1.,model.binary_brightness_factor)):
        a = amplitude*factor
        for k in range(len(means)):
            mean = means[k][None]*a[:,None]; cov = covariance[k][None]*a[:,None,None]**2+np.diag(error**2)[None]
            # Grid blocking keeps high-order refinements below a fixed memory bound.
            for start in range(0,len(a),64):
                stop = min(start+64,len(a))
                result[b,k,start:stop],bound[b,k,start:stop] = bounded_selected_probability(
                    mean[start:stop],cov[start:stop],lower,calibration_mag,support=support,draws=draws,seed=seed)
    return result,bound

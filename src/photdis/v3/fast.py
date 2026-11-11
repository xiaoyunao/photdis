"""Exact distance-dependent Gaussian evaluation by spectral factorization.

One eigendecomposition per SED kernel replaces a Cholesky decomposition at every
distance. This changes neither the likelihood nor the retained mixture branches.
The direct native-band implementation remains the independent numerical oracle.
"""
import numpy as np
from .likelihood import flux_loglikelihood


def spectral_flux_loglikelihood(flux, error, mask, transmission, template_flux,
                               template_covariance, mu, *, calibration_mag=None,
                               upper_limit=None, system='DECam_DR11',
                               template_system='DECam_DR11', dust_sigma_ebv=0.,
                               extinction_coefficients=None):
    mask = np.asarray(mask, bool)
    if upper_limit is not None and np.asarray(upper_limit, bool)[mask].any():
        return flux_loglikelihood(flux, error, mask, transmission, template_flux,
            template_covariance, mu, calibration_mag=calibration_mag,
            upper_limit=upper_limit, system=system, template_system=template_system,
            dust_sigma_ebv=dust_sigma_ebv, extinction_coefficients=extinction_coefficients)
    if system != template_system: raise ValueError('Photometric system mismatch')
    if not mask.any(): raise ValueError('No measured bands')
    f = np.asarray(flux, float)[mask]; e = np.asarray(error, float)[mask]
    tr = np.asarray(transmission, float)[mask]
    if not (np.isfinite(f).all() and np.isfinite(e).all() and (e > 0).all()
            and np.isfinite(tr).all() and (tr > 0).all()):
        raise ValueError('Invalid measured native flux, error or transmission')
    if calibration_mag is not None: tr *= 10.**(-.4*np.asarray(calibration_mag)[mask])
    sed = np.atleast_2d(template_flux)[:, mask]
    covariance = np.asarray(template_covariance)[:, mask][:, :, mask]
    if not np.isfinite(sed).all() or not np.isfinite(covariance).all():
        raise ValueError('Unsupported SED coordinates')
    factor = tr/e
    mean = sed*factor
    v = covariance*factor[None, :, None]*factor[None, None, :]
    if dust_sigma_ebv < 0 or not np.isfinite(dust_sigma_ebv):
        raise ValueError('Invalid per-object dust uncertainty')
    if dust_sigma_ebv:
        if extinction_coefficients is None: raise ValueError('Explicit extinction coefficients required')
        d = .4*np.log(10)*dust_sigma_ebv*mean*np.asarray(extinction_coefficients)[mask]
        v = v+d[:, :, None]*d[:, None, :]
    eigen, vectors = np.linalg.eigh(v)
    if (eigen < -1e-10*np.maximum(abs(eigen).max(axis=1), 1.)[:, None]).any():
        raise ValueError('Non-positive intrinsic covariance')
    eigen = np.maximum(eigen, 0.)
    projected_flux = np.einsum('kij,i->kj', vectors, f/e)
    projected_mean = np.einsum('kij,ki->kj', vectors, mean)
    amplitude = 10.**(-.4*(np.asarray(mu)-10.))
    variance = 1+eigen[:, None, :]*amplitude[None, :, None]**2
    delta = projected_flux[:, None, :]-projected_mean[:, None, :]*amplitude[None, :, None]
    return -.5*((delta**2/variance+np.log(variance)).sum(axis=-1)
                +2*np.log(e).sum()+len(f)*np.log(2*np.pi))

"""Explicit physical extinctions and fixed-model scale diagnostics."""
import numpy as np
from photdis.cluster_rgb_lower import mean, slope
from photdis.cluster_rgb_conditional import residual_measurement_cov

LEGACY_R = np.array([3.214, 2.165, 1.211])


def ccm_r(wavelength_angstrom, rv=3.1):
    """CCM89 A(lambda)/E(B-V), IR/optical domain (equations 2/3)."""
    x = 1e4 / np.asarray(wavelength_angstrom, float)
    if np.any((x < .3) | (x > 3.3)):
        raise ValueError('Only CCM IR/optical wavelengths are supported')
    y = x - 1.82
    a = np.polynomial.polynomial.polyval(y, [1,.17699,-.50447,-.02427,
                                             .72085,.01979,-.77530,.32999])
    b = np.polynomial.polynomial.polyval(y, [0,1.41338,2.28305,1.07233,
                                             -5.38434,-.62251,5.30260,-2.09002])
    a = np.where(x < 1.1, .574*x**1.61, a)
    b = np.where(x < 1.1, -.527*x**1.61, b)
    return rv*a+b


def band_extinction(wave, throughput, ebv, rv=3.1, temperature=4500):
    """Photon-counting passband integration, explicitly approximate Planck SED."""
    wave = np.asarray(wave, float)
    # B_lambda times lambda (photon weighting); arbitrary normalization cancels.
    weight = np.asarray(throughput)*wave**-4/np.expm1(1.438776877e8/(wave*temperature))
    integrate = getattr(np, 'trapezoid', np.trapz)
    ratio = integrate(weight*10**(-.4*ebv*ccm_r(wave,rv)),wave)/integrate(weight,wave)
    return float(-2.5*np.log10(ratio))


def change_scale(g0, colors, old_a, new_a, mu, dmu):
    """Undo old A, apply new A; reference M changes with the RR modulus too."""
    shift = np.asarray(old_a)-np.asarray(new_a)
    new_g = np.asarray(g0)+shift[...,0]
    delta_color = np.stack((shift[...,0]-shift[...,1],
                            shift[...,1]-shift[...,2]), axis=-1)
    return new_g, np.asarray(colors)+delta_color, new_g-(mu+dmu)


def color_geometry(model, mg, feh, colors, errors):
    residual = colors-mean(model,mg,feh)
    tangent = slope(model,mg)
    cov = residual_measurement_cov(*errors.T,tangent,2)+np.asarray(model['intrinsic'])
    chol = np.linalg.cholesky(cov)
    whitened = np.linalg.solve(chol,residual[...,None])[...,0]
    direction = np.linalg.solve(chol,tangent[...,None])[...,0]
    direction /= np.linalg.norm(direction,axis=1)[:,None]
    along = np.sum(whitened*direction,axis=1)
    perpendicular = -direction[:,1]*whitened[:,0]+direction[:,0]*whitened[:,1]
    return residual, along, perpendicular


def reference_covariance(stat, systematic, rho):
    """Each marginal keeps its published budget; correlation adds no new noise."""
    stat, systematic = np.asarray(stat), np.asarray(systematic)
    cov = rho*np.outer(systematic,systematic)
    cov[np.diag_indices_from(cov)] += stat**2+(1-rho)*systematic**2
    return cov

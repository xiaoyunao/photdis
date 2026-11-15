"""Supported SED component weights and line-of-sight priors for LF sensitivity."""
import numpy as np
from scipy.special import ndtr

from photdis.v3.likelihood import trapezoid_weights


MR_EDGES = np.array([-4.,-3.,-2.,-1.,0.,1.,2.,3.])
FEH_EDGES = np.array([-2.38,-1.6,-1.,-.45,.39])
GIANT = 1


def empirical_weights(model):
    joint = model.weights[:,None]*model.classes
    return joint/joint.sum(axis=0)


def giant_luminosity_weights(model,slope,feh_mean,feh_sigma):
    """Replace mass in occupied M_r/FeH bins, preserving sub-bin SED structure."""
    base = empirical_weights(model)
    r_flux = model.means[:,1]
    mr = np.full(len(r_flux),np.nan)
    positive = r_flux>0
    # The fitted intrinsic flux is tabulated at 1 kpc, or distance modulus 10.
    mr[positive] = 12.5-2.5*np.log10(r_flux[positive])
    feh = model.means[:,-1]
    i = np.searchsorted(MR_EDGES,mr,side='right')-1
    j = np.searchsorted(FEH_EDGES,feh,side='right')-1
    n,m = len(MR_EDGES)-1,len(FEH_EDGES)-1
    active = (i>=0)&(i<n)&(j>=0)&(j<m)
    cell = np.ravel_multi_index((i[active],j[active]),(n,m))
    existing = np.bincount(cell,weights=base[active,GIANT],minlength=n*m)
    occupied = existing>0
    if not occupied.any():
        raise ValueError('The fitted giant mixture has no kernels in the LF envelope')
    luminosity = np.diff(10.**(slope*MR_EDGES)) if slope else np.diff(MR_EDGES)
    luminosity = luminosity/luminosity.sum()
    metallicity = np.diff(ndtr((FEH_EDGES-feh_mean)/feh_sigma))
    metallicity /= metallicity.sum()
    requested = np.outer(luminosity,metallicity).ravel()
    target = np.zeros_like(existing)
    target[occupied] = existing.sum()*requested[occupied]/requested[occupied].sum()
    ratio = np.divide(target,existing,out=np.zeros_like(target),where=occupied)
    output = base.copy()
    output[active,GIANT] *= ratio[cell]
    np.testing.assert_allclose(output.sum(axis=0),1.,rtol=0,atol=1e-12)
    return output,dict(
        active_giant_mass=float(existing.sum()),occupied_bins=int(occupied.sum()),
        empty_bin_requested_fraction=float(requested[~occupied].sum()),
        giant_kernel_ess_before=float(1/np.sum(base[:,GIANT]**2)),
        giant_kernel_ess_after=float(1/np.sum(output[:,GIANT]**2)),
        largest_kernel_weight_ratio=float(np.max(ratio)))


def radial_profiles(mu,longitude,latitude,*,halo_slope=2.8,halo_flattening=.64):
    """Densities per distance modulus along one sightline, normalized on the grid."""
    d = 10.**((mu-10.)/5.)
    l,b = np.deg2rad([longitude,latitude])
    z = d*np.sin(b)+.025
    dr = d*np.cos(b)
    r_sun = 8.
    r = np.hypot(r_sun-dr*np.cos(l),dr*np.sin(l))
    thin = np.exp(-(r-r_sun)/2.6-np.abs(z)/.3)
    thick = .12*np.exp(-(r-r_sun)/3.6-np.abs(z)/.9)
    halo_radius = np.hypot(r,z/halo_flattening)
    halo = (halo_radius/r_sun)**(-halo_slope)
    profiles = d[None,:]**3*np.stack([thin+thick,halo])
    return profiles/(profiles@trapezoid_weights(mu))[:,None]

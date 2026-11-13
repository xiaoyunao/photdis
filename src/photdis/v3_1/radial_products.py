"""Distinct population parameters and realized-count products for radial controls."""
import numpy as np
from scipy.stats import norm
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3.simulation import SCENARIOS


def bin_integrals(density,grid):
    grid=np.asarray(grid);density=np.asarray(density)
    if grid.ndim!=1 or np.any(np.diff(grid)<=0) or density.shape[-1]!=len(grid):
        raise ValueError('Increasing grid matching density required')
    if grid[0]>=DISTANCE_MU_BOUNDARIES[0] or grid[-1]<=DISTANCE_MU_BOUNDARIES[-1]:
        raise ValueError('Grid must span reporting boundaries')
    # Integrate the same piecewise-linear density; insert boundaries only in
    # reporting, keeping the inference grid uniform for narrow Gaussian priors.
    augmented=np.unique(np.r_[grid,DISTANCE_MU_BOUNDARIES])
    hi=np.clip(np.searchsorted(grid,augmented,side='right'),1,len(grid)-1);lo=hi-1
    fraction=(augmented-grid[lo])/(grid[hi]-grid[lo])
    values=density[...,lo]*(1-fraction)+density[...,hi]*fraction
    edges=np.r_[0,np.searchsorted(augmented,DISTANCE_MU_BOUNDARIES),len(augmented)-1]
    return np.stack([np.einsum('...m,m->...',values[...,lo:hi+1],trapezoid_weights(augmented[lo:hi+1]))
        for lo,hi in zip(edges[:-1],edges[1:])],axis=-1)


def generator_joint_parameters(catalog):
    """Unselected class×distance fractions of the specified generating design.

    These are NOT observed-catalog realized counts or selected fractions.
    The source/noise entry experiment changes the latter.
    """
    scenario=catalog['scenario'];frequency=np.asarray(catalog['generated_class_weights'])
    if 'true_location_mu' in catalog:
        mean=np.array([catalog['true_location_mu']]);sigma=np.array([catalog['true_width_mu']]);weight=np.ones(1)
    elif scenario=='narrow_4kpc':mean=np.array([10+5*np.log10(4.)]);sigma=np.array([.08]);weight=np.ones(1)
    elif catalog.get('known_distance_family'):
        mean=DISTANCE_BASIS_MEAN;sigma=DISTANCE_BASIS_SIGMA;weight=np.ones(5)/5
    else:mean=np.array([11.,13.,16.5]);sigma=np.array([1.5,1.,2.]);weight=np.array([.45,.4,.15])
    output=[]
    for c in range(3):
        offset=([0.,3.,1.][c] if SCENARIOS.get(scenario,{}).get('correlated') else 0.)
        cdf=norm.cdf(DISTANCE_MU_BOUNDARIES[:,None],mean[None]+offset,sigma[None])@weight
        output.append(frequency[c]*np.diff(np.r_[0.,cdf,1.]))
    return np.array(output)

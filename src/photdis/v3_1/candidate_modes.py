"""Exact conditional candidate mode responsibilities on the saved PDF grid."""
import numpy as np
from scipy.signal import find_peaks
from scipy.special import logsumexp
from scipy.stats import norm
from photdis.v3.inference import DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA
from photdis.v3.likelihood import ContinuousPosterior, trapezoid_weights
from .gaussian_joint import interval_linear_weights
from .luminosity_log import log_luminosity_transform
from .luminosity_adjoint import log_luminosity_adjoint
from .target_prediction import marginalize_class_population
from .radial_products import bin_integrals


def population_pdf(native, grid, parameters, theta, log_weights):
    """Average object-normalized population draws, then frozen correction states."""
    parameters=np.asarray(parameters,float); log_weights=np.asarray(log_weights,float)
    if parameters.shape!=(len(theta),3,2) or log_weights.shape!=(len(theta),):
        raise ValueError('Aligned correction states required')
    # Stored log weights subtract large absolute evidences. Keep those bytes:
    # their roundoff is covered by the registered 1e-8 normalization gate.
    weight_normalization=float(logsumexp(log_weights))
    if not np.isclose(weight_normalization,0,atol=1e-8,rtol=0):
        raise ValueError('Frozen normalized state weights required')
    curves=[]; coefficients=[]; normalizations=[]; probabilities=[]; halves=[]
    for parameter,draws in zip(parameters,theta):
        transformed=np.array([log_luminosity_transform(native[c],grid,*parameter[c]) for c in range(3)])
        value=marginalize_class_population(transformed,grid,draws,np.full(len(draws),1/len(draws)))
        curves.append(value['class_logpdf']); coefficients.append(value['log_coefficient'])
        normalizations.append(value['log_normalization']); probabilities.append(value['class_distance'])
        halves.append(value['half_class_distance'])
    joint=logsumexp(np.array(curves)+log_weights[:,None,None],axis=0)
    normalization=float(logsumexp(joint+np.log(trapezoid_weights(grid))[None])); joint-=normalization
    lp=logsumexp(joint,axis=0); probability=bin_integrals(np.exp(joint),grid)
    expected=np.einsum('s,scd->cd',np.exp(log_weights),probabilities)
    half=np.einsum('s,shcd->hcd',np.exp(log_weights),halves)
    return dict(class_logpdf=joint,logpdf=lp,log_coefficient=np.array(coefficients),
        state_log_normalization=np.array(normalizations),log_normalization=np.array(normalization),
        state_weight_log_normalization=np.array(weight_normalization),
        class_distance=probability,mu=ContinuousPosterior(grid,np.exp(lp)[None],0.,{}).quantiles(),
        joint_delta=np.array(abs(probability-expected).max()),marginal_half_delta=np.array(abs(half[0]-half[1]).max()))


def basin_partition(logpdf, grid):
    """Use the existing mode_attribution peak and intervening valley convention."""
    density=np.exp(logpdf); peaks,_=find_peaks(density)
    if not len(peaks):peaks=np.array([density.argmax()])
    valleys=[a+np.argmin(density[a:b+1]) for a,b in zip(peaks[:-1],peaks[1:])]
    return grid[peaks],grid[np.asarray(valleys,int)]


def basin_joint_mass(native, grid, conditional, binary, parameters, log_weights, pdf, boundaries):
    """Return basin x binary x kernel x class mass using the exact adjoint.

    No extra trapezoidal factor is inserted: interval_linear_weights already
    represents the complete piecewise-linear integral on the true-mu grid.
    """
    native=np.asarray(native,float); conditional=np.asarray(conditional,float); binary=np.asarray(binary,float)
    if native.shape!=(2,len(conditional),len(grid)) or conditional.shape!=(len(conditional),3):
        raise ValueError('Binary x kernel likelihood and class-conditional weights required')
    if binary.shape!=(2,) or (binary<0).any() or (conditional<0).any() or not np.isclose(binary.sum(),1) or not np.allclose(conditional.sum(axis=0),1):
        raise ValueError('Normalized kernel and binary weights required')
    weights=interval_linear_weights(grid,boundaries)
    radial=norm.logpdf(grid[None],DISTANCE_BASIS_MEAN[:,None],DISTANCE_BASIS_SIGMA[:,None])
    with np.errstate(divide='ignore'):
        log_weights_bins=np.log(weights); log_conditional=np.log(conditional); log_binary=np.log(binary)
    mass=np.empty((len(weights),2,len(conditional),3))
    for c in range(3):
        # Bound memory when a PDF has many very weak local peaks; retain all.
        for start in range(0,len(weights),4):
            stop=min(start+4,len(weights)); accumulated=np.full((stop-start,len(grid)),-np.inf)
            for state,(parameter,lw) in enumerate(zip(parameters,log_weights)):
                radial_mix=logsumexp(pdf['log_coefficient'][state,c,:,None]+radial,axis=0)
                weight=log_weights_bins[start:stop]+radial_mix[None]
                weight-=pdf['state_log_normalization'][state]+pdf['log_normalization']
                adjoint=log_luminosity_adjoint(weight,grid,*parameter[c])
                accumulated=np.logaddexp(accumulated,lw+adjoint)
            value=logsumexp(native[None]+accumulated[:,None,None,:],axis=-1)
            mass[start:stop,:,:,c]=np.exp(value+log_binary[None,:,None]+log_conditional[None,None,:,c])
    return mass


def describe_basins(peaks, boundaries, grid, mass, parent):
    parent=np.asarray(parent)
    if parent.shape!=(mass.shape[2],) or not np.issubdtype(parent.dtype,np.integer) or (parent<0).any():
        raise ValueError('Nonnegative integer parent IDs required')
    bounds=np.r_[grid[0],boundaries,grid[-1]]; records=[]
    for i,(peak,lower,upper,joint) in enumerate(zip(peaks,bounds[:-1],bounds[1:],mass)):
        total=float(joint.sum())
        if total<=0:continue
        classes=joint.sum(axis=(0,1))/total; components=joint.sum(axis=(0,2))/total
        parents=np.bincount(parent,weights=components); binary=float(joint[1].sum()/total)
        records.append(dict(basin_index=i,mu=float(peak),lower=float(lower),upper=float(upper),mass=total,
            class_probability=classes.tolist(),dominant_class=int(classes.argmax()),binary_probability=binary,
            dominant_binary=int(binary>.5),parent_probability=parents.tolist(),dominant_parent=int(parents.argmax()),
            dominant_parent_probability=float(parents.max()),effective_parent_components=float(1/(parents@parents)),weak_basin=total<=.05))
    significant=[r for r in records if not r['weak_basin']]; pairs=[]
    for i,a in enumerate(significant):
        for b in significant[i+1:]:
            same=a['dominant_class']==b['dominant_class']
            pairs.append(dict(first=a['basin_index'],second=b['basin_index'],different_class=not same,
                same_class_different_binary=same and a['dominant_binary']!=b['dominant_binary'],
                same_class_different_parent=same and a['dominant_parent']!=b['dominant_parent']))
    return records,pairs

"""Native likelihood reconstruction for proper target-posterior log scores.

CDF compression is appropriate for probabilities but can erase tiny tails which
matter for conflicting reference observations. Recompute the complete native
likelihood for scoring; no pruned kernel list or CDF floor is used here.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from .inference import kernel_loglike, DISTANCE_BASIS_MEAN, DISTANCE_BASIS_SIGMA
from .likelihood import trapezoid_weights


def native_marginal(model,observation,grid,coefficient,reference_log_evidence,calibration_mag=None,bands=None):
    likelihood,_=kernel_loglike(model,observation,grid,bands=bands,calibration_mag=calibration_mag)
    radial=norm.logpdf(np.asarray(grid)[None],DISTANCE_BASIS_MEAN[:,None],DISTANCE_BASIS_SIGMA[:,None])
    weights=np.asarray(coefficient).sum(axis=2)
    with np.errstate(divide='ignore'):
        prior=logsumexp(np.log(weights)[:,:,None]+radial[None],axis=1)
    binary=np.array([1-model.binary_reference_fraction,model.binary_reference_fraction])
    logdensity=logsumexp(likelihood+np.log(binary)[:,None,None]+prior[None],axis=(0,1))-reference_log_evidence
    lognorm=logsumexp(logdensity+np.log(trapezoid_weights(grid)))
    return logdensity-lognorm,float(lognorm)


def native_joint_marginal(model,observation,grid,coefficient,reference_log_evidence,calibration_mag=None):
    likelihood,_=kernel_loglike(model,observation,grid,calibration_mag=calibration_mag)
    radial=norm.logpdf(np.asarray(grid)[None],DISTANCE_BASIS_MEAN[:,None],DISTANCE_BASIS_SIGMA[:,None])
    with np.errstate(divide='ignore'):
        prior=logsumexp(np.log(coefficient)[:,:,:,None]+radial[None,:,None,:],axis=1)
    binary=np.array([1-model.binary_reference_fraction,model.binary_reference_fraction])
    base=likelihood+np.log(binary)[:,None,None]
    by_class=np.array([logsumexp(base+prior[None,:,c,:],axis=(0,1)) for c in range(3)])-reference_log_evidence
    lognorm=logsumexp(by_class+np.log(trapezoid_weights(grid))[None])
    component=logsumexp(base,axis=0)+logsumexp(prior,axis=1)-reference_log_evidence-lognorm
    component_probability=np.exp(logsumexp(component+np.log(trapezoid_weights(grid))[None],axis=1))
    single=logsumexp(base[1]+logsumexp(prior,axis=1)+np.log(trapezoid_weights(grid))[None])-reference_log_evidence-lognorm
    return by_class-lognorm,dict(log_normalization=float(lognorm),binary_probability=float(np.exp(single)),component_probability=component_probability)

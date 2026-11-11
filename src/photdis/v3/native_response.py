"""Unpruned continuous component response for target-prior numerical audits."""
import numpy as np
from scipy.special import logsumexp
from .inference import kernel_loglike,distance_basis
from .compression import DISTANCE_MU_BOUNDARIES


def native_response(model,observation,grid,reference_log_evidence,*,calibration_mag=None,bands=None):
    grid=np.unique(np.r_[grid,DISTANCE_MU_BOUNDARIES])
    likelihood,_=kernel_loglike(model,observation,grid,calibration_mag=calibration_mag,bands=bands)
    binary=np.array([1-model.binary_reference_fraction,model.binary_reference_fraction])
    density=np.exp(logsumexp(likelihood+np.log(binary)[:,None,None],axis=0)-reference_log_evidence)
    radial=distance_basis(grid)
    group=np.searchsorted(DISTANCE_MU_BOUNDARIES,(grid[:-1]+grid[1:])/2,side='right')
    weights=np.zeros((6,len(grid)))
    for b in range(6):
        half=np.diff(grid)/2*(group==b);weights[b,:-1]+=half;weights[b,1:]+=half
    return np.stack([density@(radial*w).T for w in weights],axis=2)

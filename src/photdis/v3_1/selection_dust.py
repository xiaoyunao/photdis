"""Match entry integration to the likelihood's mean-linearized local dust term."""
import copy
import numpy as np


def entry_model_with_local_dust(model, transmission, ebv, dust_sigma_ebv=None):
    trans=np.asarray(transmission,float); ebv=float(ebv)
    sigma=.1*max(ebv,0.) if dust_sigma_ebv is None else float(dust_sigma_ebv)
    if trans.shape!=(6,) or not np.isfinite(ebv) or not np.isfinite(sigma) or sigma<0:
        raise ValueError('Finite six-band dust context required')
    valid=np.isfinite(trans)&(trans>0); coefficient=np.zeros(6)
    if ebv>0: coefficient[valid]=-2.5*np.log10(trans[valid])/ebv
    elif sigma>0: raise ValueError('Measured extinction coefficients require positive EBV')
    result=copy.deepcopy(model)
    # This is an integration-only covariance. Do not save it as a fitted SED.
    perturbation=.4*np.log(10)*sigma*model.means[:,:6]*coefficient[None]
    result.covariances[:,:6,:6] += perturbation[:,:,None]*perturbation[:,None,:]
    return result

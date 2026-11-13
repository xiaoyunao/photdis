"""Conditional transfer of the fixed-kernel class/radial population posterior."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3.likelihood import ContinuousPosterior,trapezoid_weights
from .gaussian_joint import interval_linear_weights
from .radial_products import bin_integrals


def marginalize_class_population(loglike,grid,theta,weights):
    """Each population draw is normalized separately before the frozen cut mixture.

    loglike is the native class likelihood after the registered luminosity
    operation, on the TRUE mu coordinate. This function does not fit or update
    population weights with new photometry. All coefficients/density tails are
    retained in log space; replacing theta by its mean would be a different model.
    """
    ll,g,t,w=map(lambda a:np.asarray(a,float),(loglike,grid,theta,weights))
    if ll.shape!=(3,len(g)) or t.ndim!=3 or t.shape[1:]!=(3,5) or w.shape!=(len(t),):
        raise ValueError('Aligned class curves and shared class/radial draws required')
    if np.isnan(ll).any() or np.isposinf(ll).any() or not np.isfinite(t).all() or (t<=0).any() or not np.allclose(t.sum(axis=(1,2)),1,atol=1e-12,rtol=0):
        raise ValueError('Valid log likelihood and positive normalized population draws required')
    if not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1,atol=1e-12):raise ValueError('Normalized shared draw weights required')
    radial=norm.logpdf(g[None],DISTANCE_BASIS_MEAN[:,None],DISTANCE_BASIS_SIGMA[:,None])
    bw=interval_linear_weights(g,DISTANCE_MU_BOUNDARIES)
    with np.errstate(divide='ignore'):lbw=np.log(bw);lw=np.log(w)
    logresponse=np.empty((3,5,6))
    for c in range(3):
        logresponse[c]=logsumexp(ll[c][None,None]+radial[:,None]+lbw[None],axis=-1)
    lt=np.log(t);evidence=logsumexp(lt+logsumexp(logresponse,axis=-1)[None],axis=(1,2))
    if not np.isfinite(evidence).all():raise ValueError('Unsupported population draw')
    coefficient=logsumexp(lw[:,None,None]+lt-evidence[:,None,None],axis=0)
    logjoint=ll+logsumexp(coefficient[:,:,None]+radial[None],axis=1)
    normalization=float(logsumexp(logjoint+np.log(trapezoid_weights(g))[None]));logjoint-=normalization
    draw_probability=np.exp(logsumexp(lt[:,:,:,None]+logresponse[None],axis=2)-evidence[:,None,None])
    mean=np.einsum('s,scb->cb',w,draw_probability);density=np.exp(logjoint);probability=bin_integrals(density,g)
    lp=logsumexp(logjoint,axis=0);q=ContinuousPosterior(g,np.exp(lp)[None],0.,{}).quantiles()
    halves=[]
    for select in (np.arange(len(w))%2==0,np.arange(len(w))%2==1):
        if w[select].sum()==0:halves=[];break
        halves.append(np.einsum('s,scb->cb',w[select]/w[select].sum(),draw_probability[select]))
    return dict(logpdf=lp,class_logpdf=logjoint,mu=q,class_distance=probability,draw_probability=draw_probability,
        log_evidence_per_draw=evidence,log_coefficient=coefficient,log_normalization=normalization,
        joint_delta=float(abs(probability-mean).max()),marginal_half_delta=float(abs(halves[0]-halves[1]).max()) if halves else np.inf)

"""Matched native likelihood with individually normalized shared-prior draws."""
import numpy as np
from scipy.special import logsumexp
from photdis.v3.inference import distance_basis
from photdis.v3.likelihood import ContinuousPosterior,trapezoid_weights
from photdis.v3.target import joint_coefficients,joint_draw_probabilities
from photdis.v3_1.radial_products import bin_integrals


def geographic_fields(ra,dec,cell,bounds):
    """Old field ownership by actual geometry; new cells have disjoint IDs.

    bounds maps original field index to [RAlo,DEClo,RAhi,DEChi]. Earlier
    original fields own overlaps, matching original catalog ownership order.
    Never equate a reference cell's numeric ID with an application field ID.
    """
    ra,dec,cell=np.asarray(ra),np.asarray(dec),np.asarray(cell)
    if ra.shape!=dec.shape or ra.shape!=cell.shape or not np.isfinite(ra).all() or not np.isfinite(dec).all():
        raise ValueError('Matching finite coordinate and cell arrays required')
    if (cell<0).any() or any(k>=10000 or k<0 for k in bounds):raise ValueError('Disjoint valid field namespaces required')
    result=10000+cell.astype('i8');assigned=np.zeros(len(ra),bool)
    for field, (lo,de,hi,dh) in sorted(bounds.items()):
        inside=((ra>=lo)&(ra<=hi)) if lo<=hi else ((ra>=lo)|(ra<=hi))
        inside&=(dec>=de)&(dec<=dh)&~assigned
        result[inside]=field;assigned|=inside
    return result,assigned


def marginalize_native(model,grid,binary_kernel_loglike,theta,beta,weights,*,batch=256):
    """Exact grid quadrature with all shared draws; no mean-prior plug-in.

    Input likelihood has binary x kernel x grid shape; class-conditional SED
    tilts and class/radial probabilities come from each corresponding draw.
    Output retains binary/kernel/class attribution and full log-density tails.
    """
    grid=np.asarray(grid);lk=np.asarray(binary_kernel_loglike);weights=np.asarray(weights,float)
    if lk.shape!=(2,len(model.weights),len(grid)):raise ValueError('Native binary/kernel/grid likelihood required')
    if weights.ndim!=1 or len(weights)<2 or (weights<0).any() or not np.isclose(weights.sum(),1):
        raise ValueError('Normalized nonnegative shared weights required')
    if len(theta)!=len(weights) or len(beta)!=len(weights):raise ValueError('Shared draw dimensions differ')
    binary=np.array([1-model.binary_reference_fraction,model.binary_reference_fraction])
    with np.errstate(divide='ignore'):base=lk+np.log(binary)[:,None,None]
    offset=float(np.max(base));density=np.exp(logsumexp(base,axis=0)-offset)
    radial=distance_basis(grid)
    response=bin_integrals(density[:,None,:]*radial[None],grid)
    average=np.zeros((len(model.weights),5,3));probability=np.empty((len(weights),3,6));evidence=np.empty(len(weights))
    for start in range(0,len(weights),batch):
        stop=min(start+batch,len(weights));coef=joint_coefficients(model,theta[start:stop],beta[start:stop])
        e=np.einsum('kd,skd->s',response.sum(axis=2),coef.sum(axis=3))
        if not np.isfinite(e).all() or (e<=0).any():raise ValueError('Unsupported shared draw')
        evidence[start:stop]=e
        average+=np.einsum('s,skdc->kdc',weights[start:stop]/e,coef)
        probability[start:stop]=joint_draw_probabilities(response[None],coef)[:,0]
    with np.errstate(divide='ignore'):
        prior=logsumexp(np.log(average)[:,:,:,None]+np.log(radial)[None,:,None,:],axis=1)
    logjoint=base[:,:,None,:]-offset+prior[None]
    lognormalization=float(logsumexp(logjoint+np.log(trapezoid_weights(grid))[None,None,None]))
    logjoint-=lognormalization
    posterior=ContinuousPosterior(grid,np.exp(logjoint).reshape(-1,len(grid)),0.,{})
    mean=np.einsum('s,scb->cb',weights,probability)
    half=[]
    for select in (np.arange(len(weights))%2==0,np.arange(len(weights))%2==1):
        w=weights[select]
        if w.sum()==0:half=[];break
        half.append(np.einsum('s,scb->cb',w/w.sum(),probability[select]))
    return dict(posterior=posterior,full_logpdf=logsumexp(logjoint,axis=(0,1,2)),
        draw_probability=probability,class_distance=mean,coefficient=average,
        log_evidence_per_draw=np.log(evidence)+offset,
        log_normalization=lognormalization,
        marginal_half_delta=float(abs(half[0]-half[1]).max()) if half else np.inf,
        joint_delta=float(abs(posterior.distance_bins().reshape(2,len(model.weights),3,6).sum(axis=(0,1))-mean).max()))

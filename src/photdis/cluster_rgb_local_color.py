"""Local color information: condition AFTER integrating shared nuisance.

The joint posterior includes a disclosed local uniform distance prior. Dividing
its marginal distance density by B(mu) yields the conditional-color posterior;
the joint evidence is an irrelevant common constant, including for atom weights.
"""
from copy import deepcopy
import numpy as np
from scipy.special import logsumexp, roots_hermitenorm, ndtr
from photdis.cluster_rgb_lower import mean, slope
from photdis.cluster_rgb_shared_distance import log_normal_interval
from photdis.cluster_rgb_shared_sampling import CollapsedTarget


def brightness_log_marginal(mu,g,sg,shared_sigma,order=128):
    nodes,weights=roots_hermitenorm(order)
    terms=[]
    for t,w in zip(nodes,weights):
        m=np.asarray(g)[:,None]+t*shared_sigma-np.asarray(mu)[None,:]
        ll=log_normal_interval((.5-m)/sg[:,None],(3.-m)/sg[:,None])
        terms.append(ll.sum(axis=0)-len(g)*np.log(2.5)+np.log(w/np.sqrt(2*np.pi)))
    return logsumexp(terms,axis=0)


def edge_crossing_bound(mu,g,sg,shared_sigma):
    """Union bound, not an independent-member probability approximation."""
    m=np.asarray(g)[:,None]-np.asarray(mu)[None,:]
    sigma=np.hypot(sg,shared_sigma)[:,None]
    probability=ndtr((.5-m)/sigma)+ndtr((m-3.)/sigma)
    return np.minimum(1.,probability.sum(axis=0))


def expand_abundance_atoms(models,atoms):
    out=[];labels=[]
    for j,z in enumerate(atoms):
        for model in models:
            shifted=deepcopy(model);beta=np.asarray(model['beta']).copy()
            beta[:,:-1]+=beta[:,-1,None]*(z+1.7)
            shifted['beta']=beta.tolist();out.append(shifted);labels.append(j)
    return out,np.asarray(labels)


class LocalColorTarget(CollapsedTarget):
    """Same target, reversible model jumps along the distance/color ridge."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        gradient=np.mean([slope(m,np.array([1.75]))[0] for m in self.models],axis=0)
        self.offset=(self.pivot-self.pivot.mean(axis=0))@gradient/max(gradient@gradient,1e-12)

    def model_proposal(self,x,old,new):
        out=x.copy();out[0]+=self.offset[new]-self.offset[old]
        before=mean(self.models[old],np.array([self.pivot_g-x[0]]),self.feh)[0]
        after=mean(self.models[new],np.array([self.pivot_g-out[0]]),self.feh)[0]
        out[1:]+=after-before
        return out


def component_bin_mass(edges,center,sigma,component,n_components):
    """Rao-Blackwell bins retaining component evidence and common grey errors.

    Each component's masses divide by the TOTAL number of joint draws, not its
    occupancy. Normalizing separately would erase the relative model evidence.
    """
    center=np.asarray(center).ravel();sigma=np.asarray(sigma).ravel()
    component=np.asarray(component).ravel()
    out=np.zeros((n_components,len(edges)-1))
    for start in range(0,len(center),256):
        c=center[start:start+256,None];s=sigma[start:start+256,None]
        ll=log_normal_interval((edges[None,:-1]-c)/s,(edges[None,1:]-c)/s)
        norm=log_normal_interval((edges[0]-c)/s,(edges[-1]-c)/s)
        mass=np.exp(ll-norm)
        for j in range(n_components):
            out[j]+=mass[component[start:start+256]==j].sum(axis=0)
    return out/len(center)


def conditional_mass(joint_mass,log_brightness):
    with np.errstate(divide='ignore'):
        lp=np.log(joint_mass)-log_brightness
    return np.exp(lp-logsumexp(lp))


def weighted_joint(component_mass,weights):
    # Sampling used equal atom probabilities; retain posterior occupancy weights.
    out=np.asarray(weights)*len(weights)@component_mass
    return out/out.sum()


def summarize_local(edges,p):
    p=np.asarray(p,float);p=p/p.sum();centers=(edges[:-1]+edges[1:])/2
    q=np.interp([.025,.16,.5,.84,.975],np.r_[0.,np.cumsum(p)],edges)
    prior=np.diff(edges)/(edges[-1]-edges[0]);positive=p>0
    left=float(p[centers<edges[0]+.05].sum());right=float(p[centers>edges[-1]-.05].sum())
    mode=float(centers[np.argmax(p)])
    return dict(q=q.tolist(),width68=float(q[3]-q[1]),width95=float(q[4]-q[0]),
        kl_nats=float(np.sum(p[positive]*np.log(p[positive]/prior[positive]))),
        edge_mass_left=left,edge_mass_right=right,mode=mode,
        edge_flag=bool(max(left,right)>.15 or mode<edges[0]+.05 or mode>edges[-1]-.05))

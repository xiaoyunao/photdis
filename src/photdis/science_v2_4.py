"""Frozen-PDF scientific diagnostics, with explicit populations and dependence."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def distance_bin_mass(pdf, edges, g, distance_edges):
    """Integrate class×effective-M or marginal PDFs; preserve channel mass."""
    p=np.asarray(pdf,float);g=np.asarray(g,float)
    de=np.asarray(distance_edges,float)
    with np.errstate(divide='ignore'):mu=10+5*np.log10(de)
    out=[]
    for lo,hi in zip(mu[:-1],mu[1:]):
        left=g-hi;right=g-lo
        fraction=np.maximum(0,np.minimum(edges[1:],right[:,None])-np.maximum(edges[:-1],left[:,None]))/np.diff(edges)
        out.append(np.sum(p*fraction.reshape((len(g),)+(1,)*(p.ndim-2)+(len(edges)-1,)),axis=-1))
    return np.stack(out,axis=-1)


def pdf_reference_shape(pdf, edges, g, ref):
    """Operational peak descriptions, not physical class/cause assignments."""
    p=np.asarray(pdf,float);target=np.asarray(g)-np.asarray(ref)
    width=np.diff(edges);center=(edges[:-1]+edges[1:])/2
    frac=np.maximum(0,np.minimum(edges[1:],target[:,None]+1)-np.maximum(edges[:-1],target[:,None]-1))/width
    mass=np.sum(p*frac,axis=1);mode=[];near_peak=[];peak_count=[]
    for q,t,gg in zip(p,target,g):
        density=gaussian_filter1d(q/width,.1/np.median(width))
        maxima=find_peaks(density,prominence=.01*density.max())[0]
        mode.append(gg-center[np.argmax(density)])
        near_peak.append(bool(np.any(abs(center[maxima]-t)<1)))
        peak_count.append(len(maxima))
    valid=np.isfinite(ref)
    return dict(mass_within1=np.where(valid,mass,np.nan),mode_mu=np.asarray(mode),
                peak_near_reference=np.asarray(near_peak)&valid,peak_count=np.asarray(peak_count))


def cluster_mean_interval(values, clusters, repeats=1000, seed=20261111):
    """Resample entire clusters; repeated control identities move together."""
    v=np.asarray(values,float);c=np.asarray(clusters);good=np.isfinite(v)
    v=v[good];c=c[good]
    if not len(v):return dict(mean=None,low=None,high=None,clusters=0)
    _,ix=np.unique(c,return_inverse=True);k=ix.max()+1
    means=np.bincount(ix,weights=v);counts=np.bincount(ix)
    if k<2:return dict(mean=float(v.mean()),low=None,high=None,clusters=k)
    rng=np.random.default_rng(seed);choice=rng.integers(k,size=(repeats,k))
    boot=means[choice].sum(axis=1)/counts[choice].sum(axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(mean=float(v.mean()),low=float(lo),high=float(hi),clusters=int(k))


def conditional_prior(cal_y,cal_cells,eval_cells,strength=30.):
    """Only calibration labels enter this r/sky-only population benchmark."""
    counts=np.bincount(np.asarray(cal_y,int),minlength=3).astype(float)+.5
    prior=counts/counts.sum();out=np.tile(prior,(len(eval_cells),1))
    for cell in np.unique(eval_cells):
        use=np.asarray(cal_cells)==cell
        out[np.asarray(eval_cells)==cell]=(np.bincount(np.asarray(cal_y)[use],minlength=3)+strength*prior)/(use.sum()+strength)
    return prior,out


def weighted_scores(prob,y,weights):
    p=np.asarray(prob,float);y=np.asarray(y,int);w=np.asarray(weights,float);w=w/w.sum()
    one=np.eye(3)[y];log=np.log(np.maximum(p[np.arange(len(y)),y],1e-12))
    ece=0.
    for c in range(3):
        bins=np.minimum((p[:,c]*10).astype(int),9)
        for b in range(10):
            u=bins==b
            if u.any():ece+=abs(np.sum(w[u]*(p[u,c]-one[u,c])))/3
    return dict(log_score=float(w@log),brier=float(w@np.sum((p-one)**2,axis=1)),ece=float(ece))

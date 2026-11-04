"""Identity-safe helpers for the broader v1.4 reference assembly."""
import numpy as np
from scipy.spatial import cKDTree


def unit_vectors(ra, dec):
    r, d = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)))


def epoch_positions(ra, dec, epoch, pmra, pmdec, target_epoch=2016.):
    """Tangent-vector propagation, including RA wrap and cos(dec) convention.

    Missing/zero epochs or proper motions retain static positions with an explicit
    unknown-epoch flag. Those matches receive a stricter separation threshold.
    """
    ra, dec, epoch, pmra, pmdec = [np.asarray(v, float) for v in (ra,dec,epoch,pmra,pmdec)]
    valid = (np.isfinite(epoch)&(epoch>1900)&(epoch<2100)&np.isfinite(pmra)&np.isfinite(pmdec)
             &(abs(pmra)<1e5)&(abs(pmdec)<1e5))
    r,d = np.deg2rad(ra),np.deg2rad(dec)
    east = np.column_stack((-np.sin(r),np.cos(r),np.zeros(len(r))))
    north = np.column_stack((-np.sin(d)*np.cos(r),-np.sin(d)*np.sin(r),np.cos(d)))
    dt = np.where(valid,target_epoch-epoch,0)
    v = unit_vectors(ra,dec) + (east*np.nan_to_num(pmra)[:,None]+north*np.nan_to_num(pmdec)[:,None])*dt[:,None]*np.pi/(180*3.6e6)
    v /= np.linalg.norm(v,axis=1)[:,None]
    return v, valid


def positional_join(source, candidate):
    """Conservative one-to-one, epoch-aware match; ambiguity is never ranked away.

    All distinct native candidates within2arcsec invalidate a match. Accept at
    1arcsec if both epochs are known, otherwise0.5arcsec. Inputs must have unique
    source identities and native ls_ids before calling.
    """
    a, ak = epoch_positions(*[source[k] for k in ('ra','dec','epoch','pmra','pmdec')])
    b, bk = epoch_positions(*[candidate[k] for k in ('ra','dec','epoch','pmra','pmdec')])
    index=np.full(len(a),-1,'i8');sep=np.full(len(a),np.nan);unknown=np.zeros(len(a),bool)
    ambiguous=np.zeros(len(a),bool)
    if not len(b):return index,sep,unknown,ambiguous
    tree=cKDTree(b);radius=2*np.sin(np.deg2rad(2/3600)/2)
    for i,neighbors in enumerate(tree.query_ball_point(a,radius)):
        ambiguous[i]=len(neighbors)>1
        if len(neighbors)!=1:continue
        j=neighbors[0];dist=np.rad2deg(2*np.arcsin(np.clip(np.linalg.norm(a[i]-b[j])/2,0,1)))*3600
        known=ak[i]&bk[j]
        if dist<=(1. if known else .5):index[i]=j;sep[i]=dist;unknown[i]=not known
    good=index>=0
    u,c=np.unique(index[good],return_counts=True)
    collision=good&np.isin(index,u[c>1]);index[collision]=-1;ambiguous[collision]=True
    return index,sep,unknown,ambiguous


def representative_rows(identity, primary, snr):
    """Evidence-independent primary/highest-SNR/original-order representative."""
    order=np.lexsort((np.arange(len(identity)),-np.nan_to_num(snr,nan=-np.inf),-np.asarray(primary,dtype=int),identity))
    return order[np.r_[True,identity[order][1:]!=identity[order][:-1]]]


def duplicate_mask(key):
    u,c=np.unique(key,return_counts=True)
    return np.isin(key,u[c>1])

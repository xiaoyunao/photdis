"""Mixtures of counts with one shared calibration state per observed catalog."""
import numpy as np


def mix_counts(results,weights,state_ids):
    w=np.asarray(weights,float);states=np.asarray(state_ids,'i8')
    if len(results)!=len(w) or states.shape!=w.shape or len(np.unique(states))!=len(states):
        raise ValueError('One unique state identity per conditional result required')
    if not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1,atol=1e-12):
        raise ValueError('Normalized calibration weights required')
    n=int(results[0]['n']);means=np.array([r['mean'] for r in results]);mean=w@means
    if any(int(r['n'])!=n for r in results):raise ValueError('Identical observed catalog required')
    conditional=np.einsum('s,sij->ij',w,np.array([r['conditional_covariance'] for r in results]))
    within=np.einsum('s,sij->ij',w,np.array([r['shared_covariance'] for r in results]))
    deviation=means-mean;between=(deviation.T*w)@deviation;shared=within+between
    probability=np.concatenate([q*np.asarray(r['draw_weights']) for q,r in zip(w,results)])
    counts=np.concatenate([r['count_draws'] for r in results]);conditional_means=np.concatenate([r['conditional_means'] for r in results])
    state=np.concatenate([np.full(len(r['draw_ids']),s,'i8') for s,r in zip(states,results)])
    within_id=np.concatenate([r['draw_ids'] for r in results]);assert counts.dtype==np.int64 and (counts.sum(axis=1)==n).all()
    np.testing.assert_allclose(probability@conditional_means,mean,atol=1e-8,rtol=1e-12)
    return dict(n=n,mean=mean,covariance=conditional+shared,conditional_covariance=conditional,
        shared_covariance=shared,within_state_shared_covariance=within,between_state_covariance=between,
        draw_weights=probability,count_draws=counts,conditional_means=conditional_means,
        global_draw_ids=np.arange(len(probability),dtype='i8'),state_ids=state,within_state_draw_ids=within_id,
        state_effective_sample_size=float(1/(w@w)),draw_effective_sample_size=float(1/(probability@probability)))


def realized_count_intervals(result):
    """Weighted discrete intervals from retained conditional categorical draws.

    The covariance remains the Rao-Blackwell result above; sampled percentiles
    are Monte Carlo diagnostics, not exact mixture quantiles.
    """
    n=int(result['n']);counts=result['count_draws'];weight=result['draw_weights'];intervals=[];half_delta=0.
    halves=np.asarray(result['within_state_draw_ids'])%2
    for k in range(counts.shape[1]):
        mass=np.bincount(counts[:,k],weights=weight,minlength=n+1);cdf=np.cumsum(mass);cdf[-1]=1.
        intervals.append(np.searchsorted(cdf,[.025,.5,.975]))
        hc=[]
        for j in (0,1):
            select=halves==j;den=weight[select].sum()
            if den<=0:hc=[];half_delta=np.inf;break
            hc.append(np.cumsum(np.bincount(counts[select,k],weights=weight[select]/den,minlength=n+1)))
        if hc:half_delta=max(half_delta,float(abs(hc[0]-hc[1]).max()))
    return np.array(intervals,dtype='i8'),half_delta

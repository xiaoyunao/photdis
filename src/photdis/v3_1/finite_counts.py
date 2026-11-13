"""Exact marginal count distributions conditional on a finite shared response set."""
import numpy as np


def finite_count_marginals(probabilities,weights):
    """Mix conditional Poisson-binomial marginals; do not mix stars first.

    Returned component marginals are not independent. Their joint covariance
    must be computed from the same shared categorical states separately.
    """
    p,w=np.asarray(probabilities,float),np.asarray(weights,float)
    if p.ndim!=3 or w.shape!=(len(p),) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=2),1,atol=1e-10,rtol=0):
        raise ValueError('State/object/component probabilities required')
    if not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1):raise ValueError('Normalized shared weights required')
    s,n,k=p.shape;mass=np.zeros((s,k,n+1));mass[:,:,0]=1
    for row in range(n):
        q=p[:,row,:,None];updated=mass*(1-q)
        updated[:,:,1:]+=mass[:,:,:-1]*q;mass=updated
    result=np.einsum('s,skn->kn',w,mass)
    if not np.allclose(result.sum(axis=1),1,atol=1e-12,rtol=0):raise ValueError('Count probability conservation failed')
    return result

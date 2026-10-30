"""Stage-restricted RGB controls; v1 functions/results stay frozen."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.special import logsumexp
from photdis.cluster_rgb_conditional import (
    basis, residual_measurement_cov, stable_cov, log_gaussian)

KNOTS = np.array([.5, 1.25, 2., 3.])


def design(m, z, kind):
    m = np.asarray(m, float)
    if kind == 'quadratic':
        return basis(m, z)
    # Knot ordinates with linear interpolation, linear continuation for
    # perturbed training labels outside the fixed nominal support.
    j = np.clip(np.searchsorted(KNOTS, m, side='right')-1, 0, len(KNOTS)-2)
    t = (m-KNOTS[j])/(KNOTS[j+1]-KNOTS[j])
    x = np.zeros(m.shape+(5,))
    flat = x.reshape(-1,5)
    flat[np.arange(m.size),j.ravel()] = (1-t).ravel()
    flat[np.arange(m.size),j.ravel()+1] = t.ravel()
    x[..., -1] = np.broadcast_to(z, m.shape)+1.7
    return x


def mean(model, m, z):
    return design(m,z,model['kind']) @ np.asarray(model['beta']).T


def slope(model, m):
    b = np.asarray(model['beta'])
    m = np.asarray(m, float)
    if model['kind'] == 'quadratic':
        return b[:,1]+2*(m[...,None]-1)*b[:,2]
    j = np.clip(np.searchsorted(KNOTS,m,side='right')-1,0,len(KNOTS)-2)
    return ((b[:,j+1]-b[:,j])/(KNOTS[j+1]-KNOTS[j])).T


def fit(a, nc, kind, ridge=4., floor=.01):
    x = design(a['mg'],a['feh'],kind)
    y = a['color'][:,:nc]
    _, ix, counts = np.unique(a['cluster'],return_inverse=True,return_counts=True)
    w = 1./(len(counts)*counts[ix])
    d = x.shape[1]
    b = np.linalg.lstsq(x,y,rcond=None)[0].T
    cov = np.eye(nc)*.03**2
    constraints = []
    if kind == 'monotone':
        c = np.zeros((3*nc,d*nc))
        for k in range(nc):
            for j in range(3):
                c[3*k+j,d*k+j] = -1
                c[3*k+j,d*k+j+1] = 1
        constraints = [LinearConstraint(c,-np.inf,0)]
    for _ in range(8):
        mod = dict(kind=kind,beta=b)
        v = residual_measurement_cov(a['sg'],a['sr'],a['sz'],slope(mod,a['mg']),nc)
        inv = np.linalg.inv(v+cov)
        h = np.einsum('n,njk,na,nb->jakb',w,inv,x,x).reshape(d*nc,d*nc)
        rhs = np.einsum('n,njk,nk,na->ja',w,inv,y,x).ravel()
        for k in range(nc):
            h[k*d+d-1,k*d+d-1] += ridge
        if constraints:
            result = minimize(lambda q: .5*q@h@q-rhs@q,b.ravel(),
                              jac=lambda q:h@q-rhs,constraints=constraints,
                              method='SLSQP',options={'ftol':1e-10,'maxiter':500})
            if not result.success:
                raise RuntimeError(result.message)
            b = result.x.reshape(nc,d)
        else:
            b = np.linalg.solve(h,rhs).reshape(nc,d)
        r = y-x@b.T
        cov = stable_cov((r.T*w)@r-np.einsum('n,nij->ij',w,v),floor)
    return dict(kind=kind,beta=b.tolist(),intrinsic=cov.tolist())


def probabilities(models, grid, z, zw, colors, errors):
    """Batch uniform-grid posterior; g/colour shared error as in v1."""
    colors, errors = np.asarray(colors), np.asarray(errors)
    nc = colors.shape[1]
    total = np.full((len(colors),len(grid)),-np.inf)
    for model in models:
        s = slope(model,grid)
        v = residual_measurement_cov(errors[:,0,None],errors[:,1,None],
                                     errors[:,2,None],s[None,:,:],nc)
        mixture = model.get('contamination', {'weights':[1.], 'scales':[1.]})
        for zz, weight in zip(z,zw):
            r = colors[:,None,:]-mean(model,grid,zz)[None,:,:]
            for cw,scale in zip(mixture['weights'],mixture['scales']):
                ll = log_gaussian(r,v+scale*np.asarray(model['intrinsic']))+np.log(cw*weight/len(models))
                total = np.logaddexp(total,ll)
    return np.exp(total-logsumexp(total,axis=1)[:,None])


def summarize(grid, prob, reference_mg, bin_reference=None):
    """Point-reference diagnostics; delta_mu = reference_Mg - median_Mg."""
    cdf = np.cumsum(prob,axis=1)
    q = np.array([[grid[min(np.searchsorted(row,t),len(grid)-1)]
                   for t in (.025,.16,.5,.84,.975)] for row in cdf])
    ref = np.asarray(reference_mg)
    residual = ref-q[:,2]
    step = grid[1]-grid[0]
    nearest = np.argmin(abs(ref[:,None]-grid),axis=1)
    logp = np.log(np.maximum(prob[np.arange(len(ref)),nearest]/step,1e-300))
    outside = (ref < grid[0]-step/2) | (ref >= grid[-1]+step/2)
    logp[outside] = np.log(1e-300)
    def stat(mask):
        r = residual[mask]
        if not len(r): return {'n':0}
        return dict(n=len(r),outside_support_n=int(outside[mask].sum()),median=float(np.median(r)),
                    nmad=float(1.4826*np.median(abs(r-np.median(r)))),
                    tail=float(np.mean(abs(r)>.5)),
                    cover68=float(np.mean((ref[mask]>=q[mask,1])&(ref[mask]<=q[mask,3]))),
                    cover95=float(np.mean((ref[mask]>=q[mask,0])&(ref[mask]<=q[mask,4]))),
                    width68=float(np.median(q[mask,3]-q[mask,1])),
                    log_density=float(np.mean(logp[mask])))
    bins = ref if bin_reference is None else np.asarray(bin_reference)
    return dict(overall=stat(np.ones(len(ref),bool)),
                bins={f'{lo:.1f}_{lo+.5:.1f}':stat((bins>=lo)&(bins<lo+.5))
                      for lo in np.arange(.5,3,.5)}),q,residual


def fit_contamination(a, nc, floor=.01, ridge=4.):
    """Exact fixed 95% narrow + 5% broad Gaussian residual likelihood.

    Broad covariance is 25 times narrow intrinsic covariance; measurement
    covariance is added separately to both. Every source remains in the fit.
    """
    initial=fit(a,nc,'monotone',floor=floor,ridge=ridge)
    x=design(a['mg'],a['feh'],'monotone');d=x.shape[1]
    y=a['color'][:,:nc]
    _,ix,counts=np.unique(a['cluster'],return_inverse=True,return_counts=True)
    w=1./(len(counts)*counts[ix])
    cov=np.asarray(initial['intrinsic'])
    l=np.linalg.cholesky(stable_cov(cov-floor**2*np.eye(nc),1e-5))
    ij=np.tril_indices(nc)
    start=np.r_[np.asarray(initial['beta']).ravel(),l[ij]]
    c=np.zeros((3*nc,len(start)))
    for k in range(nc):
        for j in range(3):c[3*k+j,d*k+j]=-1;c[3*k+j,d*k+j+1]=1
    def unpack(theta):
        b=theta[:nc*d].reshape(nc,d)
        chol=np.zeros((nc,nc));chol[ij]=theta[nc*d:]
        return b,chol@chol.T+floor**2*np.eye(nc)
    def objective(theta):
        b,s=unpack(theta)
        mod=dict(kind='monotone',beta=b)
        v=residual_measurement_cov(a['sg'],a['sr'],a['sz'],slope(mod,a['mg']),nc)
        r=y-x@b.T
        ll=np.logaddexp(np.log(.95)+log_gaussian(r,v+s),
                        np.log(.05)+log_gaussian(r,v+25*s))
        return -w@ll+.5*ridge*np.sum(b[:,-1]**2)
    opt=minimize(objective,start,method='SLSQP',constraints=[LinearConstraint(c,-np.inf,0)],
                 options={'ftol':1e-9,'maxiter':400})
    if not opt.success:raise RuntimeError('contamination fit: '+opt.message)
    b,s=unpack(opt.x)
    return dict(kind='monotone',beta=b.tolist(),intrinsic=s.tolist(),
                contamination={'weights':[.95,.05],'scales':[1.,25.]},
                optimization_iterations=int(opt.nit),objective=float(opt.fun))

"""Four-component gate learned from independent luminosity evidence.

Frozen v1 experts and catalogs are deliberately not changed.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from .baseline_v0 import features


def gate_objective(theta, x, likelihood, labels, sample_weight, l2=.001,
                   class_weight=1.):
    """Weighted luminosity likelihood plus reliable-class supervision.

    Likelihoods must be cross-fit predictions. A common per-row likelihood
    rescaling has no effect on the fitted gate or its gradient.
    """
    b=theta.reshape(x.shape[1],4)
    logits=x@b;logp=logits-logsumexp(logits,axis=1,keepdims=True)
    p=np.exp(logp);li=np.maximum(likelihood,1e-300)
    logjoint=logp+np.log(li);logmix=logsumexp(logjoint,axis=1)
    responsibility=np.exp(logjoint-logmix[:,None])
    loss=-logmix;grad=p-responsibility
    known=labels>=0;ii=np.flatnonzero(known)
    loss[known]-=class_weight*logp[ii,labels[known]]
    grad[known]+=class_weight*p[known]
    grad[ii,labels[known]]-=class_weight
    w=np.asarray(sample_weight,float);w=w/w.sum()
    # The intercept is unpenalized; softmax's common shift is irrelevant.
    penalty=b.copy();penalty[0]=0
    value=float(w@loss+.5*l2*np.sum(penalty**2))
    gradient=x.T@(w[:,None]*grad)+l2*penalty
    return value,gradient.ravel()


class EvidenceGate:
    """Observed-color weights; unknown labels never specify residual weight."""
    def __init__(self,dim,l2=.001,class_weight=1.,maxiter=1000):
        self.dim=dim;self.l2=l2;self.class_weight=class_weight;self.maxiter=maxiter

    def fit(self,mags,likelihood,labels,sample_weight):
        color=features(mags,self.dim)
        self.scaler=StandardScaler().fit(color,sample_weight=sample_weight)
        self.poly=PolynomialFeatures(2,include_bias=True)
        x=self.poly.fit_transform(self.scaler.transform(color))
        if not np.isfinite(likelihood).all() or np.any(likelihood<0):
            raise ValueError('Finite nonnegative cross-fit evidence required')
        if np.any(likelihood.sum(axis=1)<=0):raise ValueError('Empty evidence')
        start=np.zeros((x.shape[1],4))
        result=minimize(gate_objective,start.ravel(),
            args=(x,likelihood,labels,sample_weight,self.l2,self.class_weight),
            jac=True,method='L-BFGS-B',options=dict(maxiter=self.maxiter,ftol=1e-10,gtol=1e-7))
        self.coef=result.x.reshape(x.shape[1],4)
        self.optimization=dict(success=bool(result.success),message=str(result.message),
            iterations=int(result.nit),objective=float(result.fun),
            gradient_max=float(np.max(np.abs(result.jac))))
        return self

    def predict_proba(self,mags):
        x=self.poly.transform(self.scaler.transform(features(mags,self.dim)))
        logits=x@self.coef
        return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def transfer_unsupported(weights,branch_supported):
    """Preserve total mass; residual is a distribution, not a physical class."""
    w=np.asarray(weights,float).copy()
    lost=w[:,:3]*(~np.asarray(branch_supported,bool))
    w[:,:3]-=lost;w[:,3]+=lost.sum(axis=1)
    return w,lost.sum(axis=1)


def oracle_weights(labels,branch_supported,unknown_fraction,mode):
    labels=np.asarray(labels);known=(labels>=0)&(labels<3)
    w=np.zeros((len(labels),4));w[:,3]=1
    ii=np.flatnonzero(known);w[ii,3]=0;w[ii,labels[known]]=1
    if mode=='direct':return w,np.zeros(len(w))
    if mode=='full':
        u=np.asarray(unknown_fraction);w[:,:3]*=1-u[:,None]
        w[known,3]=u[known]
    elif mode!='support':raise ValueError(mode)
    return transfer_unsupported(w,branch_supported)


def integrate(model,mags,errors,ids,variants,labels=None,gate=None,draws=128):
    """Shared-noise comparisons with explicit fallback accounting.

    The evidence gate is conditioned on the observed colors, like its training
    objective. Expert likelihoods marginalize shared band errors. Original v1
    natural/balanced gates retain their original per-draw definition.
    """
    from .baseline_v1 import (support,qmc_noise,shift_pdf,quantiles_interpolated)
    from .baseline_v0 import summaries,distance_quantiles
    wide=model.wide;n=len(mags)
    bs=np.column_stack([support(model.branches[k],mags) if k in model.branches
                        else np.zeros(n,bool) for k in range(3)])
    broad_support=support(wide,mags)
    pdf={k:np.zeros((n,len(wide.centers))) for k in variants}
    weights={k:np.zeros((n,4)) for k in variants}
    fallback={k:np.zeros(n) for k in variants};clsum=np.zeros((n,4))
    fixed=None
    if 'evidence' in variants:
        if gate is None:raise ValueError('Fitted independent-evidence gate required')
        fixed=transfer_unsupported(gate.predict_proba(mags),bs)
    errors=np.nan_to_num(errors,nan=0,posinf=0)
    for draw in range(draws):
        delta=qmc_noise(ids,draw)*errors;x=features(mags+delta,model.dim)
        broad,cl=wide.distribution(x);clsum+=cl
        experts={k:b.distribution(x)[0] for k,b in model.branches.items()}
        gp=model.gate.predict_proba(x)
        for name in variants:
            lost=np.zeros(n)
            if name=='wide':p=broad;w=cl
            else:
                if name in ('oracle','oracle_support','oracle_full'):
                    if labels is None:raise ValueError('Diagnostic oracle requires labels')
                    mode={'oracle':'direct','oracle_support':'support','oracle_full':'full'}[name]
                    w,lost=oracle_weights(labels,bs,cl[:,3],mode)
                    # Missing trained expert cannot supply even diagnostic mass.
                    for k in range(3):
                        if k not in experts:w[:,3]+=w[:,k];w[:,k]=0
                elif name in ('natural','balanced'):
                    g=gp.copy()
                    if name=='balanced':g/=np.maximum(model.frequencies,1e-10)
                    g/=g.sum(axis=1)[:,None]
                    w,lost=transfer_unsupported(np.column_stack([g*(1-cl[:,3,None]),cl[:,3]]),bs)
                elif name=='evidence':w,lost=fixed
                else:raise ValueError(name)
                p=w[:,3,None]*broad
                for k,bp in experts.items():p+=w[:,k,None]*bp
            pdf[name]+=shift_pdf(p,delta[:,0],wide.edges)
            weights[name]+=w;fallback[name]+=lost
    cl=clsum/draws;known=1-cl[:,3]
    conditional=np.divide(cl[:,:3],known[:,None],out=np.zeros((n,3)),where=known[:,None]>0)
    entropy=-np.sum(conditional*np.log(np.maximum(conditional,1e-300)),axis=1)
    result={}
    for name,p in pdf.items():
        p/=p.sum(axis=1)[:,None]
        q=quantiles_interpolated(p,wide.edges);_,modes,amb=summaries(p,wide.centers)
        mu,distance=distance_quantiles(mags[:,0],q)
        result[name]=dict(pdf=p.astype('f4'),mu=mu,distance=distance,modes=modes,
            ambiguous=amb,supported=broad_support,weights=weights[name]/draws,
            support_fallback=fallback[name]/draws,branch_support=bs,
            unknown_fraction=cl[:,3],known_condmax=conditional.max(axis=1),
            known_entropy=entropy)
    return result

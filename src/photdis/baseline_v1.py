"""Versioned population mixtures and reproducible photometry integration.

The v0 class and serialized models are intentionally unchanged.
"""
from functools import lru_cache
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from .baseline_v0 import (HistogramQRF, features, band_noise, summaries,
                          Q, distance_quantiles)


def quantiles_interpolated(pdf, edges):
    """Inverse of a piecewise-uniform histogram CDF, including empty bins."""
    c=np.cumsum(pdf,axis=1,dtype=float);c/=c[:,-1,None]
    idx=np.column_stack([np.argmax(c>=q,axis=1) for q in Q])
    before=np.take_along_axis(np.column_stack([np.zeros(len(c)),c]),idx,axis=1)
    after=np.take_along_axis(c,idx,axis=1)
    frac=np.divide(Q-before,after-before,out=np.zeros_like(before),where=after>before)
    return edges[idx]+np.clip(frac,0,1)*(edges[idx+1]-edges[idx])


@lru_cache(maxsize=1)
def sobol_bits():
    return (qmc.Sobol(5,scramble=True,seed=20261103).random_base2(10)*2**32).astype(np.uint32)


def qmc_noise(ids, draw, replicate=0):
    """Digital-shift randomized Sobol; identity/chunk and band invariant."""
    with np.errstate(over='ignore'):
        x=np.asarray(ids,dtype=np.uint64)[:,None]+np.arange(5,dtype=np.uint64)*np.uint64(0x9e3779b97f4a7c15)+np.uint64(replicate+1)*np.uint64(0xd1b54a32d192ed03)
        x=(x^(x>>30))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>27))*np.uint64(0x94d049bb133111eb);x=x^(x>>31)
    u=((x>>32).astype(np.uint32)^sobol_bits()[draw]).astype(float)
    return ndtri((u+.5)/2**32)


def support(model,mags):
    x=features(mags,model.dim)
    dist=model.nn.kneighbors((x-model.mean)/model.scale,n_neighbors=10)[0][:,-1]
    return (dist<=model.radius)&np.all((x>=model.lo)&(x<=model.hi),axis=1)


def align_histogram(branch,wide):
    """All fixed 0.1mag bins share an integer origin; preserve exact leaf mass."""
    idx=np.searchsorted(wide.edges,branch.centers,side='right')-1
    assert np.allclose(wide.centers[idx],branch.centers)
    branch.tables=[np.pad(h,((0,0),(idx[0],len(wide.centers)-idx[-1]-1))) for h in branch.tables]
    branch.edges=wide.edges.copy();branch.centers=wide.centers.copy()
    return branch


class PopulationModel:
    def __init__(self,wide):
        self.wide=wide;self.dim=wide.dim;self.edges=wide.edges;self.centers=wide.centers

    def fit(self,mags,y,labels,seed=20261103):
        known=labels>=0
        self.frequencies=np.bincount(labels[known],minlength=3)/known.sum()
        self.gate=make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),
            LogisticRegression(C=1.,max_iter=1500,random_state=seed))
        self.gate.fit(features(mags[known],self.dim),labels[known])
        assert np.array_equal(self.gate[-1].classes_,np.arange(3))
        self.branches={}
        for k in range(3):
            take=labels==k
            if take.sum()<100:continue
            h=HistogramQRF(self.dim,seed=seed+k,n_estimators=128,min_samples_leaf=32,
                           max_depth=20,bin_width=self.wide.bin_width)
            self.branches[k]=align_histogram(h.fit(mags[take],y[take],labels[take]),self.wide)
        return self


def shift_pdf(pdf,delta_g,edges):
    n=len(pdf);step=edges[1]-edges[0]
    cumulative=np.column_stack([np.zeros(n),np.cumsum(pdf,axis=1)])
    e=(edges[None,:]+delta_g[:,None]-edges[0])/step
    lower=np.floor(e).astype(int);fraction=e-lower
    c0=np.take_along_axis(cumulative,np.clip(lower,0,len(edges)-1),axis=1)
    c1=np.take_along_axis(cumulative,np.clip(lower+1,0,len(edges)-1),axis=1)
    c=c0*(1-fraction)+c1*fraction;c[:,0]=0;c[:,-1]=1
    out=np.maximum(np.diff(c,axis=1),0)
    return out/out.sum(axis=1)[:,None]


def integrate(model,mags,errors,ids,draws=128,method='sobol',replicate=0,
              variants=('wide',),oracle_labels=None,keep_pdf=True):
    """Combine each noisy color's weights and luminosities before marginalizing.

    oracle_labels can affect only the oracle result, never deployed variants.
    Unsupported branch weight is transferred to the broad frozen model.
    """
    wide=model.wide if isinstance(model,PopulationModel) else model
    n=len(mags);pdf={k:np.zeros((n,len(wide.centers)),np.float64) for k in variants}
    weights={k:np.zeros((n,4),float) for k in variants}
    in_wide=support(wide,mags)
    need_branch=any(k!='wide' for k in variants)
    in_branch={k:support(b,mags) for k,b in model.branches.items()} if need_branch else {}
    err=np.nan_to_num(errors,nan=0,posinf=0)
    for d in range(draws):
        z=band_noise(ids,d) if method=='iid' else qmc_noise(ids,d,replicate)
        delta=z*err;x=features(mags+delta,wide.dim)
        broad,cl=wide.distribution(x)
        candidates={'wide':broad};cw={'wide':cl}
        if need_branch:
            gp=model.gate.predict_proba(x)
            branch_pdf={k:b.distribution(x)[0] for k,b in model.branches.items()}
            for name in ('natural','balanced'):
                if name not in variants:continue
                g=gp if name=='natural' else gp/np.maximum(model.frequencies,1e-10)
                g=g/g.sum(axis=1)[:,None]
                w=g*(1-cl[:,3,None]);u=cl[:,3].copy()
                for k in range(3):
                    ok=in_branch.get(k,np.zeros(n,bool))
                    u+=w[:,k]*(~ok);w[:,k]*=ok
                p=u[:,None]*broad
                for k,bp in branch_pdf.items():p+=w[:,k,None]*bp
                candidates[name]=p;cw[name]=np.column_stack([w,u])
            if 'oracle' in variants:
                if oracle_labels is None:raise ValueError('Oracle requires diagnostic labels')
                p=broad.copy();ow=np.zeros((n,4));ow[:,3]=1
                for k,bp in branch_pdf.items():
                    use=oracle_labels==k;p[use]=bp[use];ow[use,3]=0;ow[use,k]=1
                candidates['oracle']=p;cw['oracle']=ow
        for name in variants:
            pdf[name]+=shift_pdf(candidates[name],delta[:,0],wide.edges)
            weights[name]+=cw[name]
    result={}
    for name,p in pdf.items():
        p/=draws;p/=p.sum(axis=1)[:,None]
        q=quantiles_interpolated(p,wide.edges)
        _,modes,amb=summaries(p,wide.centers)
        mu,dist=distance_quantiles(mags[:,0],q)
        result[name]=dict(mu=mu,distance=dist,modes=modes,ambiguous=amb,
                          weights=weights[name]/draws,supported=in_wide)
        if keep_pdf:result[name]['pdf']=p.astype('f4')
    return result

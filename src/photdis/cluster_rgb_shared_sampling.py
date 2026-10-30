"""Posterior-directed integration of the same frozen shared-distance model.

Gaussian nuisance shifts collapse exactly to (delta_g, delta_color). Conditional
delta_g is integrated analytically; only distance-minus-delta_g, color shifts and
the frozen discrete calibration realization are sampled. No reference mu is used.
"""
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from photdis.cluster_rgb_lower import KNOTS, mean, slope
from photdis.cluster_rgb_shared_distance import log_normal_interval


def effective_covariance(model, rband, sigma_e, sigma_zero, sigma_feh):
    nc = len(model['beta'])
    transform = np.array([[1.,0,0],[1,-1,0],[0,1,-1]])[:nc+1]
    band_cov = sigma_zero**2*np.eye(3)+sigma_e**2*np.outer(rband,rband)
    cov = transform@band_cov@transform.T
    abundance = np.r_[0.,-np.asarray(model['beta'])[:,-1]]
    cov += sigma_feh**2*np.outer(abundance,abundance)
    return cov


class PreparedLikelihood:
    """Cache fixed covariances for fast exact scalar-distance evaluation."""
    def __init__(self, model, g, colors, errors, feh):
        self.g, self.colors = np.asarray(g),np.asarray(colors)
        nc = colors.shape[1]
        transform = np.array([[1.,0,0],[1,-1,0],[0,1,-1]])[:nc+1]
        measure = np.einsum('ab,nb,cb->nac',transform,errors**2,transform)
        mix = model.get('contamination',{'weights':[1.],'scales':[1.]})
        self.components=[]
        for weight,scale in zip(mix['weights'],mix['scales']):
            cov=measure.copy();cov[:,1:,1:]+=scale*np.asarray(model['intrinsic'])
            inv=np.linalg.inv(cov);logdet=np.linalg.slogdet(cov)[1]
            for lo,hi in zip(KNOTS[:-1],KNOTS[1:]):
                gradient=slope(model,np.array([(lo+hi)/2]))[0]
                intercept=np.r_[0.,mean(model,np.array([lo]),feh)[0]-gradient*lo]
                direction=np.r_[1.,gradient]
                invdir=np.einsum('nij,j->ni',inv,direction)
                a=invdir@direction
                prefactor=(-.5*nc*np.log(2*np.pi)-.5*logdet-.5*np.log(a)
                           -np.log(2.5)+np.log(weight))
                self.components.append((lo,hi,intercept,direction,inv,invdir,a,prefactor))

    def rows(self,theta,dc,n=None):
        n=len(self.g) if n is None else n
        y0=np.column_stack((self.g[:n]-theta,self.colors[:n]+dc))
        total=np.full(n,-np.inf)
        for lo,hi,intercept,direction,inv,invdir,a,prefactor in self.components:
            y=y0-intercept
            mhat=np.sum(y*invdir[:n],axis=1)/a[:n]
            residual=y-mhat[:,None]*direction
            chi=np.einsum('ni,nij,nj->n',residual,inv[:n],residual)
            mass=log_normal_interval((lo-mhat)*np.sqrt(a[:n]),(hi-mhat)*np.sqrt(a[:n]))
            total=np.logaddexp(total,prefactor[:n]-.5*chi+mass)
        return total


class CollapsedTarget:
    def __init__(self, models, g, colors, errors, feh, covariances, n, mu_bounds=(10.,22.)):
        self.models=models;self.n=n;self.bounds=mu_bounds
        self.pivot_g=float(np.median(g[:n]));self.feh=feh
        self.likelihoods=[PreparedLikelihood(m,g,colors,errors,feh) for m in models]
        self.pivot=np.array([mean(m,np.array([1.75]),feh)[0] for m in models])
        self.inv=[];self.logdet=[];self.gain=[];self.conditional_sigma=[]
        for cov in covariances:
            cc=cov[1:,1:];inv=np.linalg.inv(cc)
            gain=cov[0,1:]@inv
            variance=cov[0,0]-gain@cov[1:,0]
            if variance<=0:raise ValueError('Degenerate shared grey-offset covariance')
            self.inv.append(inv);self.logdet.append(np.linalg.slogdet(cc)[1])
            self.gain.append(gain);self.conditional_sigma.append(np.sqrt(variance))
        self.inv=np.array(self.inv);self.logdet=np.array(self.logdet)
        self.gain=np.array(self.gain);self.conditional_sigma=np.array(self.conditional_sigma)

    def __call__(self,x,k):
        theta,dc=x[0],x[1:]
        prior=-.5*(dc@self.inv[k]@dc+self.logdet[k]+len(dc)*np.log(2*np.pi))
        center=theta+self.gain[k]@dc;sig=self.conditional_sigma[k]
        mass=log_normal_interval((self.bounds[0]-center)/sig,(self.bounds[1]-center)/sig)
        return float(self.likelihoods[k].rows(theta,dc,self.n).sum()+prior+mass)

    def model_proposal(self,x,old,new):
        proposed=x.copy()
        proposed[1:]+=self.pivot[new]-self.pivot[old]
        return proposed

    def conditional_mu(self,x,k):
        center=x[...,0]+np.sum(self.gain[k]*x[...,1:],axis=-1)
        return center,self.conditional_sigma[k]

    def distance_proposal(self,x,shift,k):
        """Reversible unit-Jacobian shear along the RGB distance/color ridge."""
        proposed=x.copy();proposed[0]+=shift
        m=np.array([self.pivot_g-x[0],self.pivot_g-proposed[0]])
        predicted=mean(self.models[k],m,self.feh)
        proposed[1:]+=predicted[1]-predicted[0]
        return proposed


def metropolis_chains(target,start,model_count,*,chains=4,warmup=3000,draws=12000,seed=103015):
    dim=len(start);samples=np.empty((chains,draws,dim));indices=np.empty((chains,draws),int)
    logps=np.empty((chains,draws));accept=[]
    for chain in range(chains):
        rng=np.random.default_rng(seed+chain*1009)
        x=np.asarray(start,float).copy();x[0]+=(chain-(chains-1)/2)*.15
        k=chain*model_count//chains;lp=target(x,k)
        cov=np.diag(np.r_[.07,np.full(dim-1,.012)]**2)
        chol=np.linalg.cholesky(cov);history=[];accepted=np.zeros(3);attempts=np.zeros(3)
        for j in range(warmup+draws):
            u=rng.random()
            kind=1 if u<.25 else (2 if u<.45 and hasattr(target,'distance_proposal') else 0)
            if kind==1:
                kk=int(rng.integers(model_count));xx=target.model_proposal(x,k,kk)
            elif kind==2:
                kk=k;xx=target.distance_proposal(x,rng.normal(0,.3),k)
            else:
                kk=k;xx=x+chol@rng.normal(size=dim)
            ll=target(xx,kk)
            if np.log(rng.random())<ll-lp:
                x,k,lp=xx,kk,ll
                if j>=warmup:accepted[kind]+=1
            if j<warmup:
                history.append(x.copy())
                if j>=999 and (j+1)%200==0:
                    recent=np.array(history[-2000:])
                    cov=(2.38**2/dim)*np.cov(recent,rowvar=False)
                    cov+=np.diag(np.r_[.005,np.full(dim-1,.0005)]**2)
                    chol=np.linalg.cholesky(cov)
            else:
                attempts[kind]+=1
                samples[chain,j-warmup]=x;indices[chain,j-warmup]=k;logps[chain,j-warmup]=lp
        accept.append(np.divide(accepted,attempts,out=np.zeros(3),where=attempts>0).tolist())
    return samples,indices,logps,accept


def _split(x):
    half=x.shape[1]//2
    return np.concatenate([x[:,:half],x[:,-half:]],axis=0)


def _normal_ranks(x):
    return ndtri((rankdata(x.ravel()).reshape(x.shape)-.375)/(x.size+.25))


def _rhat(x):
    n=x.shape[1];within=np.var(x,axis=1,ddof=1).mean()
    between=n*np.var(x.mean(axis=1),ddof=1)
    return float(np.sqrt(((n-1)*within/n+between/n)/within))


def _ess(x):
    m,n=x.shape;centered=x-x.mean(axis=1)[:,None]
    fft=np.fft.rfft(centered,n=2*n,axis=1)
    acov=np.fft.irfft(fft*np.conj(fft),n=2*n,axis=1)[:,:n]/n
    within=np.var(x,axis=1,ddof=1).mean()
    total=(n-1)*within/n+np.var(x.mean(axis=1),ddof=1)
    if total<=0:return float(m*n)
    rho=1-(within-acov.mean(axis=0))/total;rho[0]=1
    pairs=[]
    for j in range(0,n-1,2):
        pair=rho[j]+rho[j+1]
        if pair<0:break
        pairs.append(min(pair,pairs[-1]) if pairs else pair)
    return float(min(m*n,m*n/max(1.,-1+2*sum(pairs))))


def chain_diagnostics(x):
    """Rank-normalized/folded split Rhat and bulk/tail ESS (Vehtari et al.)."""
    split=_split(np.asarray(x));ranked=_normal_ranks(split)
    folded=_normal_ranks(abs(split-np.median(split)))
    lo,hi=np.quantile(split,[.05,.95])
    return dict(rhat=max(_rhat(ranked),_rhat(folded)),bulk_ess=_ess(ranked),
                tail_ess=min(_ess((split<=lo).astype(float)),_ess((split>=hi).astype(float))))


def conditional_density(grid,center,sigma,bounds=(10.,22.)):
    """Rao-Blackwellized mu-bin mass; the shared grey component stays analytic."""
    step=grid[1]-grid[0];edges=np.r_[grid-step/2,grid[-1]+step/2]
    # Prior bounds remain exact at the two outer grid cells.
    edges=np.clip(edges,*bounds)
    out=np.zeros(len(grid))
    center,sigma=np.asarray(center).ravel(),np.asarray(sigma).ravel()
    for first in range(0,len(center),512):
        c=center[first:first+512,None];s=sigma[first:first+512,None]
        mass=np.diff(ndtr((edges[None,:]-c)/s),axis=1)
        norm=ndtr((bounds[1]-c)/s)-ndtr((bounds[0]-c)/s)
        out+=(mass/norm).sum(axis=0)
    return out/out.sum()

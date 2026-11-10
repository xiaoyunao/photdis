"""Fixed heteroscedastic extreme-deconvolution density experiment.

Latent coordinates are (g-r, r-z, Mg). Class is auxiliary, never an input.
Bovy, Hogg & Roweis (2011), arXiv:0905.2979, identity projection case.
"""
import numpy as np
from scipy.special import logsumexp, ndtr
from sklearn.mixture import GaussianMixture


def observation_covariance(errors, reference_sigma):
    e = np.asarray(errors, float)[:, :3]**2
    s = np.asarray(reference_sigma, float)
    if not np.isfinite(e).all() or not np.isfinite(s).all() or np.any(s < 0):
        raise ValueError('Finite nonnegative observation uncertainties required')
    out = np.zeros((len(e), 3, 3))
    out[:, 0, 0] = e[:, 0]+e[:, 1]
    out[:, 1, 1] = e[:, 1]+e[:, 2]
    out[:, 2, 2] = e[:, 0]+s*s
    out[:, 0, 1] = out[:, 1, 0] = -e[:, 1]
    out[:, 0, 2] = out[:, 2, 0] = e[:, 0]
    return out


def inverse3(t):
    """Vectorized symmetric positive-definite3x3 inverse and log determinant."""
    a,b,c,d,e,f = t[:,0,0],t[:,0,1],t[:,0,2],t[:,1,1],t[:,1,2],t[:,2,2]
    aa,bb,cc,dd,ee,ff = d*f-e*e,c*e-b*f,b*e-c*d,a*f-c*c,b*c-a*e,a*d-b*b
    det = a*aa+b*bb+c*cc
    if np.any(det <= 0) or not np.isfinite(det).all():
        raise ValueError('Non-positive covariance determinant')
    inv = np.empty_like(t)
    inv[:,0,0]=aa;inv[:,0,1]=inv[:,1,0]=bb;inv[:,0,2]=inv[:,2,0]=cc
    inv[:,1,1]=dd;inv[:,1,2]=inv[:,2,1]=ee;inv[:,2,2]=ff
    return inv/det[:,None,None], np.log(det)


def conditional_latent(y, errors, mean, covariance):
    inverse, logdet = inverse3(errors+covariance)
    gain = np.einsum('ab,nbc->nac', covariance, inverse)
    delta = y-mean
    posterior_mean = mean+np.einsum('nab,nb->na', gain, delta)
    posterior_cov = covariance-np.einsum('nab,bc->nac', gain, covariance)
    log_density = -.5*(3*np.log(2*np.pi)+logdet+np.einsum('na,nab,nb->n',delta,inverse,delta))
    return posterior_mean, posterior_cov, log_density


class ErrorDeconvolvedMixture:
    def __init__(self, components=32, seed=20261110, max_iter=256, tol=1e-5):
        self.components=components; self.seed=seed; self.max_iter=max_iter; self.tol=tol
        self.floor=1e-6

    def expectation(self, y, s):
        logs=[]; means=[]; covs=[]
        for w,m,v in zip(self.weights,self.means,self.covariances):
            b,B,l=conditional_latent(y,s,m,v)
            means.append(b);covs.append(B);logs.append(l+np.log(w))
        log=np.column_stack(logs);norm=logsumexp(log,axis=1)
        return np.exp(log-norm[:,None]), np.stack(means,axis=1), np.stack(covs,axis=1), norm

    def fit(self, y, s, labels, callback=None):
        y=np.asarray(y,float);s=np.asarray(s,float);labels=np.asarray(labels)
        if not np.isfinite(y).all() or not np.isfinite(s).all(): raise ValueError('Invalid training observations')
        # Standardization affects initialization only; all EM covariances in mag².
        scale=y.std(axis=0);scale[scale==0]=1
        init=GaussianMixture(self.components,random_state=self.seed,n_init=1,max_iter=100,reg_covar=1e-6).fit(y/scale)
        self.weights=init.weights_;self.means=init.means_*scale
        self.covariances=init.covariances_*scale[None,:,None]*scale[None,None,:]
        self.history=[];self.converged=False
        for iteration in range(self.max_iter):
            count=np.zeros(self.components);first=np.zeros_like(self.means);second=np.zeros_like(self.covariances);ll=0.
            for start in range(0,len(y),4096):
                r,b,B,l=self.expectation(y[start:start+4096],s[start:start+4096])
                count+=r.sum(axis=0);first+=np.einsum('nk,nka->ka',r,b)
                second+=np.einsum('nk,nkab->kab',r,B)+np.einsum('nk,nka,nkb->kab',r,b,b);ll+=l.sum()
            ll/=len(y);self.history.append(float(ll))
            if callback is not None: callback(iteration,ll)
            if iteration>=10 and abs(ll-self.history[-2]) < self.tol*(1+abs(ll)):
                self.converged=True;break
            if np.any(count<=0): raise ValueError('Empty mixture component')
            self.weights=count/count.sum();self.means=first/count[:,None]
            v=second/count[:,None,None]-self.means[:,:,None]*self.means[:,None,:]
            eig,vec=np.linalg.eigh((v+v.transpose(0,2,1))/2)
            self.covariances=np.einsum('kab,kb,kcb->kac',vec,np.maximum(eig,self.floor),vec)
        # Auxiliary frequency includes missing-class channel; no class balancing.
        counts=np.zeros((self.components,4))
        for start in range(0,len(y),4096):
            r,_,_,_=self.expectation(y[start:start+4096],s[start:start+4096])
            c=np.where(labels[start:start+4096]<0,3,labels[start:start+4096])
            for k in range(4): counts[:,k]+=r[c==k].sum(axis=0)
        self.class_frequency=counts/counts.sum(axis=1)[:,None]
        return self

    def conditional(self, mags, errors):
        """Mixture of effective M=Mg-error_g given observed colors.

        Cov(effective_M, observed_colors)=V_Mc-(sigma_g²,0). This minus
        sign retains the SAME g perturbation in color and distance modulus.
        """
        m=np.asarray(mags,float);e=np.asarray(errors,float)[:,:3]**2
        color=np.column_stack((m[:,0]-m[:,1],m[:,1]-m[:,2]))
        means=[];variances=[];logs=[]
        for w,mu,v in zip(self.weights,self.means,self.covariances):
            a=v[0,0]+e[:,0]+e[:,1];b=v[0,1]-e[:,1];d=v[1,1]+e[:,1]+e[:,2]
            det=a*d-b*b
            if np.any(det<=0): raise ValueError('Invalid color covariance')
            delta=color-mu[:2]
            solved=np.column_stack((d*delta[:,0]-b*delta[:,1],a*delta[:,1]-b*delta[:,0]))/det[:,None]
            cross=np.column_stack((v[2,0]-e[:,0],np.full(len(e),v[2,1])))
            mean=mu[2]+np.sum(cross*solved,axis=1)
            variance=v[2,2]+e[:,0]-(d*cross[:,0]**2-2*b*cross[:,0]*cross[:,1]+a*cross[:,1]**2)/det
            if np.any(variance<=0): raise ValueError('Non-positive conditional variance')
            logs.append(np.log(w)-.5*(2*np.log(2*np.pi)+np.log(det)+np.sum(delta*solved,axis=1)))
            means.append(mean);variances.append(variance)
        log=np.column_stack(logs);weight=np.exp(log-logsumexp(log,axis=1)[:,None])
        return weight,np.column_stack(means),np.sqrt(np.column_stack(variances))

    def predict(self, mags, errors, edges):
        w,mean,sigma=self.conditional(mags,errors)
        pdf=np.zeros((len(mags),len(edges)-1));class_mass=w@self.class_frequency
        escaped=np.zeros(len(mags));q=np.zeros(len(mags))
        for k in range(self.components):
            cdf=ndtr((edges[None,:]-mean[:,k,None])/sigma[:,k,None])
            pdf+=w[:,k,None]*np.diff(cdf,axis=1)
            escaped+=w[:,k]*(cdf[:,0]+1-cdf[:,-1])
            q+=w[:,k]*ndtr((mean[:,k]-(mags[:,0]-15))/sigma[:,k])
        if np.any(escaped>1e-6): raise ValueError('PDF grid tail loss >1e-6; preserve mixture, do not silently truncate')
        pdf/=pdf.sum(axis=1)[:,None]
        return dict(pdf=pdf,weights=class_mass,qnear=q,mixture_weight=w,mixture_mean=mean,mixture_sigma=sigma,escaped_mass=escaped)

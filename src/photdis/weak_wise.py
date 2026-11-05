"""A shared joint luminosity/IR template posterior with signed-flux updates."""
import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from .baseline_v1 import qmc_noise


def colors(mags):
    return mags[:, :-1]-mags[:, 1:]


def flux_log_likelihood(observed, inverse_variance, prediction):
    """Gaussian log likelihood relative to the observational normalization."""
    scale = np.sqrt(inverse_variance)[:,None]
    predicted_snr = prediction*scale
    variance = 1 + (.05*predicted_snr)**2
    residual = observed[:,None]*scale-predicted_snr
    return -.5*(residual**2/variance + np.log(variance))


class JointWISE:
    def __init__(self, grz, absolute_g, w1w2, edges, neighbors=96):
        x = colors(grz)
        self.scale = x.std(axis=0)
        if np.any(self.scale<=0) or not np.isfinite(x).all() or len(x)<neighbors:
            raise ValueError('Joint templates require finite and varying colors')
        self.tree = cKDTree(x/self.scale)
        self.absolute_g = np.asarray(absolute_g)
        self.ratio = 10**(.4*(grz[:,2,None]-w1w2))
        self.edges = np.asarray(edges)
        self.neighbors = neighbors
        assert np.isfinite(self.ratio).all() and (self.ratio>0).all()
        assert np.isfinite(self.absolute_g).all()

    def predict(self, mags, errors, ids, flux, ivar, transmission, valid, draws=32):
        n = len(ids)
        nbins = len(self.edges)-1
        position = np.empty((n,draws,self.neighbors))
        log_opt = np.empty_like(position)
        log_ir = np.zeros((*position.shape,2))
        assert np.isfinite(mags).all() and np.isfinite(errors).all()
        for j in range(draws):
            delta = qmc_noise(ids,j)[:,:3]*errors
            noisy = mags+delta
            distance,index = self.tree.query(colors(noisy)/self.scale,k=self.neighbors)
            width = np.maximum(.03,.5*distance[:,-1,None])
            w = -.5*(distance/width)**2
            log_opt[:,j] = w-logsumexp(w,axis=1,keepdims=True)
            position[:,j] = (self.absolute_g[index]-delta[:,0,None]-self.edges[0])/(self.edges[1]-self.edges[0])-.5
            z_flux = 10**(.4*(22.5-noisy[:,2]))
            for b in range(2):
                use = valid[:,b]
                prediction = self.ratio[index[use],b]*((z_flux*transmission[:,b])[use,None])
                log_ir[use,j,:,b] = flux_log_likelihood(flux[use,b],ivar[use,b],prediction)
        p = np.clip(position.reshape(n,-1),0,nbins-1)
        lower = np.floor(p).astype(np.int64)
        frac = p-lower
        a = (np.arange(n)[:,None]*nbins+lower).ravel()
        b = (np.arange(n)[:,None]*nbins+np.minimum(lower+1,nbins-1)).ravel()
        results, ess = [], []
        for log_w in (log_opt,log_opt+log_ir[:,:,:,0],log_opt+log_ir.sum(axis=-1)):
            log_w = log_w.reshape(n,-1)
            w = np.exp(log_w-logsumexp(log_w,axis=1,keepdims=True))
            hist = np.bincount(a,(w*(1-frac)).ravel(),minlength=n*nbins)
            hist += np.bincount(b,(w*frac).ravel(),minlength=n*nbins)
            results.append(hist.reshape(n,nbins))
            ess.append(1/np.sum(w*w,axis=1))
        return np.stack(results,axis=1),np.column_stack(ess)

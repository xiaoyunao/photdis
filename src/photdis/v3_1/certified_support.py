"""Exact support certificates from the 1-Lipschitz kth-neighbor distance."""
import numpy as np
from .exact_support import ExactColorSupport


class CertifiedColorSupport(ExactColorSupport):
    """Accelerate interior decisions; ambiguous points use the original query.

    For the original standardized Euclidean metric, |d10(x)-d10(c)|<=|x-c|.
    This is a support certificate, not interpolation of the hard boundary.
    A conservative roundoff guard sends numerical boundary cases to eps=0.
    """
    def __init__(self,model,resolution=1024):
        super().__init__(model)
        if model.dim!=2 or resolution<2: raise ValueError('Two-dimensional color support required')
        self.resolution=int(resolution)
        self.lower=(np.asarray(model.lo)-model.mean)/model.scale
        upper=(np.asarray(model.hi)-model.mean)/model.scale
        self.step=(upper-self.lower)/self.resolution
        if not np.isfinite(self.step).all() or (self.step<=0).any(): raise ValueError('Positive finite standardized support box required')
        self.center_distance=np.empty((self.resolution,self.resolution))
        for start in range(0,self.resolution,32):
            ix,iy=np.meshgrid(np.arange(start,min(start+32,self.resolution)),np.arange(self.resolution),indexing='ij')
            center=self.lower+(np.c_[ix.ravel(),iy.ravel()]+.5)*self.step
            distance=self.tree.query(center,k=[10],eps=0.,workers=1)[0][:,0]
            self.center_distance[start:start+len(ix)]=distance.reshape(ix.shape)
        self.queries=0; self.certified=0; self.fallback=0

    def __call__(self,magnitudes):
        model=self.model; x=model.feature_matrix(magnitudes)
        inside=((x>=model.lo)&(x<=model.hi)).all(axis=1)
        answer=np.zeros(len(x),bool); self.queries+=len(x)
        if not inside.any(): return answer
        z=(x[inside]-model.mean)/model.scale
        index=np.clip(np.floor((z-self.lower)/self.step).astype(int),0,self.resolution-1)
        center=self.lower+(index+.5)*self.step
        radius=np.linalg.norm(z-center,axis=1)
        distance=self.center_distance[index[:,0],index[:,1]]
        guard=1e-12*(1+np.linalg.norm(z,axis=1)+np.linalg.norm(center,axis=1)+abs(model.radius)+distance)
        accept=distance+radius+guard<model.radius
        reject=distance-radius-guard>model.radius
        uncertain=~(accept|reject); self.certified+=int((accept|reject).sum()); self.fallback+=int(uncertain.sum())
        if uncertain.any(): accept[uncertain]=self.tree.query(z[uncertain],k=[10],eps=0.,workers=1)[0][:,0]<=model.radius
        answer[inside]=accept
        return answer

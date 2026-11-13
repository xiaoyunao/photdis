"""Original field shrinkage with within-class SED proportions fixed exactly."""
import numpy as np
from scipy.special import softmax
from photdis.v3.hierarchy import FieldPopulation


class FixedKernelFieldPopulation(FieldPopulation):
    """Collapse kernels before fitting class x radial weights; no SED-tilt axes.

    Removing the two unused standard-normal tilt coordinates at beta_sigma=0
    integrates an independent normalized prior to one. The retained objective
    and gradients equal the original hierarchy, without a nuisance degeneracy.
    """
    def __init__(self,response,selection,class_frequency,field,case_weights=None,
                 global_sigma=.7,field_sigma=.35):
        response,selection=np.asarray(response,float),np.asarray(selection,float)
        frequency=np.asarray(class_frequency,float);field=np.asarray(field)
        if response.ndim!=3 or response.shape!=selection.shape or response.shape[:2]!=(len(field),3): raise ValueError('Aligned object x class x radial response and selection required')
        if not np.isfinite(response).all() or (response<0).any() or not (response.sum(axis=(1,2))>0).all(): raise ValueError('Supported nonnegative finite responses required')
        if not np.isfinite(selection).all() or (selection<0).any() or (selection>1+1e-12).any() or not (selection.sum(axis=(1,2))>0).all(): raise ValueError('Supported conditional entry probabilities required')
        if frequency.shape!=(3,) or (frequency<=0).any() or not np.isclose(frequency.sum(),1): raise ValueError('Three positive reference class frequencies required')
        self.c=3;self.d=response.shape[2];self.k=self.c*self.d-1
        self.fields,owner=np.unique(field,return_inverse=True)
        weights=np.ones(len(field)) if case_weights is None else np.asarray(case_weights,float)
        if weights.shape!=(len(field),) or not np.isfinite(weights).all() or (weights<=0).any(): raise ValueError('Positive per-object design weights required')
        weights=weights/weights.mean();self.parts=[]
        for j in range(len(self.fields)):
            use=owner==j
            self.parts.append((np.ascontiguousarray(response[use].reshape(use.sum(),-1)),
                np.ascontiguousarray(selection[use].reshape(use.sum(),-1)),weights[use]))
        base=(frequency[:,None]*np.full((1,self.d),1/self.d)).ravel()
        self.center=np.log(base[:-1]/base[-1]);self.global_sigma=global_sigma;self.field_sigma=field_sigma
        if global_sigma<=0 or field_sigma<=0: raise ValueError('Positive frozen field/global prior scales required')
        self.dimension=self.k*(1+len(self.fields));self.case_weight_sum=float(weights.sum())
        self.effective_case_count=float(weights.sum()**2/(weights@weights))

    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(self.dimension,) or not np.isfinite(z).all(): raise ValueError('Finite aligned population coordinates required')
        global_logits=self.center+self.global_sigma*z[:self.k]
        local=global_logits[None]+self.field_sigma*z[self.k:].reshape(len(self.fields),self.k)
        theta=softmax(np.c_[local,np.zeros(len(local))],axis=1).reshape(-1,self.c,self.d)
        return theta,global_logits

    def objective(self,z):
        theta,_=self.decode(z);value=.5*np.dot(z,z);gradient=np.array(z,copy=True);local_gradient=[]
        for t,(response,selection,weight) in zip(theta,self.parts):
            flat=t.ravel();total=response@flat;entry=selection@flat
            if (total<=0).any() or (entry<=0).any(): raise ValueError('Zero conditional population support')
            value-=weight@(np.log(total)-np.log(entry))
            g=flat*((weight/entry)@selection-(weight/total)@response)
            g-=flat*g.sum();local_gradient.append(g[:-1])
        local_gradient=np.array(local_gradient)
        gradient[:self.k]+=self.global_sigma*local_gradient.sum(axis=0)
        gradient[self.k:]+=self.field_sigma*local_gradient.ravel()
        return float(value),gradient

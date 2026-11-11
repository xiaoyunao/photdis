"""Exact value-only field likelihood batches for numerical integration."""
import numpy as np
from scipy.special import softmax


def objective_batch(hierarchy,samples,batch_size=128):
    samples=np.asarray(samples,float)
    if samples.ndim!=2 or samples.shape[1]!=hierarchy.dimension:raise ValueError('Incorrect hierarchy coordinates')
    result=np.empty(len(samples));h=hierarchy
    for first in range(0,len(samples),batch_size):
        z=samples[first:first+batch_size];g=h.center+h.global_sigma*z[:,:h.k]
        beta=h.beta_sigma*z[:,h.k:h.k+2]
        local=g[:,None]+h.field_sigma*z[:,h.k+2:].reshape(len(z),len(h.fields),h.k)
        theta=softmax(np.concatenate([local,np.zeros((*local.shape[:2],1))],axis=2),axis=2).reshape(len(z),len(h.fields),h.c,h.d)
        prior=softmax(np.log(h.model.weights)[None,:,None]+np.log(h.model.classes)[None]+(beta@h.features.T)[:,:,None],axis=1)
        value=.5*np.sum(z*z,axis=1)
        for j,(r,a,w) in enumerate(h.matrices):
            coefficient=np.einsum('skc,scd->skd',prior,theta[:,j]).reshape(len(z),-1).T
            total=r@coefficient;entry=a@coefficient
            valid=(total>0).all(axis=0)&(entry>0).all(axis=0)
            with np.errstate(divide='ignore',invalid='ignore'):value-=w@(np.log(total)-np.log(entry))
            value[~valid]=np.inf
        result[first:first+len(z)]=value
    return result

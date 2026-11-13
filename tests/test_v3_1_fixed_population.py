import numpy as np
from test_v3_inference import model
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3_1.fixed_population import FixedKernelFieldPopulation


def test_fixed_kernel_hierarchy_equals_original_zero_tilt_and_chain_rule():
    rng=np.random.default_rng(319801);m=model()
    response=rng.lognormal(0,1,(43,2,5));selection=rng.uniform(.001,1,response.shape)
    fields=np.arange(43)%3;weights=rng.uniform(.2,4,len(fields))
    prior=m.weights[:,None]*m.classes;prior/=prior.sum(axis=0)[None]
    old=FastFieldPopulation(response,selection,m,fields,weights,beta_sigma=0.)
    new=FixedKernelFieldPopulation(np.einsum('nkd,kc->ncd',response,prior),np.einsum('nkd,kc->ncd',selection,prior),m.weights@m.classes,fields,weights)
    keep=np.r_[np.arange(old.k),np.arange(old.k+2,old.dimension)]
    for scale in (.01,1.,3.):
        z=rng.normal(0,scale,new.dimension);original=np.zeros(old.dimension);original[keep]=z
        a,ag=old.objective(original);b,bg=new.objective(z)
        np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-12)
        np.testing.assert_allclose(ag[keep],bg,atol=1e-11,rtol=1e-12)
        np.testing.assert_allclose(new.decode(z)[0],old.decode(original)[0],atol=1e-14)
    z=rng.normal(size=new.dimension);_,gradient=new.objective(z);eps=1e-5
    for i in (0,13,14,new.dimension-1):
        dz=np.zeros(new.dimension);dz[i]=eps
        finite=(new.objective(z+dz)[0]-new.objective(z-dz)[0])/(2*eps)
        np.testing.assert_allclose(gradient[i],finite,atol=2e-8,rtol=1e-7)

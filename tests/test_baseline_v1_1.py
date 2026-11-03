import numpy as np
from photdis.baseline_v1_1 import gate_objective,oracle_weights,integrate
from photdis.baseline_v1 import PopulationModel,integrate as old_integrate
from photdis.baseline_v0 import HistogramQRF


def test_joint_likelihood_gradient_and_rescaling():
    rng=np.random.default_rng(4);x=rng.normal(size=(17,5));x[:,0]=1
    li=np.exp(rng.normal(size=(17,4)));labels=np.arange(17)%4-1
    w=rng.uniform(.1,2,17);theta=rng.normal(size=20)*.1
    _,g=gate_objective(theta,x,li,labels,w);fd=[]
    for j in range(len(theta)):
        d=np.zeros_like(theta);d[j]=1e-6
        fd.append((gate_objective(theta+d,x,li,labels,w)[0]-gate_objective(theta-d,x,li,labels,w)[0])/2e-6)
    assert np.allclose(g,fd,atol=1e-8)
    _,g2=gate_objective(theta,x,li*rng.uniform(.1,10,(17,1)),labels,w)
    assert np.allclose(g,g2,atol=1e-12)


def test_oracle_fallback_mass_and_unknown_labels():
    bs=np.array([[1,1,1],[1,0,1],[1,1,1]],bool)
    a,lost=oracle_weights(np.array([1,1,-1]),bs,np.array([.8,.2,.7]),'full')
    assert np.allclose(a,[[0,.2,0,.8],[0,0,0,1],[0,0,0,1]])
    assert np.allclose(lost,[0,.8,0])
    a,_=oracle_weights(np.array([1,1,-1]),bs,np.zeros(3),'direct')
    assert np.allclose(a,[[0,1,0,0],[0,1,0,0],[0,0,0,1]])


def test_original_variants_reproduced_and_diagnostic_labels_isolated():
    rng=np.random.default_rng(8);m=rng.normal(size=(450,5));labels=np.r_[np.repeat(np.arange(3),120),np.full(90,-1)]
    y=np.where(labels<0,0,labels)*3.
    wide=HistogramQRF(2,n_estimators=2,min_samples_leaf=8).fit(m,y,labels)
    model=PopulationModel(wide).fit(m,y,labels)
    mags=m[:8];errors=np.full((8,5),.03);ids=np.arange(8)
    old=('wide','natural','balanced','oracle');allv=old+('oracle_support','oracle_full')
    a=old_integrate(model,mags,errors,ids,draws=8,variants=old,oracle_labels=np.ones(8,int))
    b=integrate(model,mags,errors,ids,allv,labels=np.ones(8,int),draws=8)
    c=integrate(model,mags,errors,ids,allv,labels=np.zeros(8,int),draws=8)
    for k in old:
        assert np.allclose(a[k]['pdf'],b[k]['pdf'],atol=2e-7)
        assert np.allclose(a[k]['mu'],b[k]['mu'],atol=2e-6)
        if k!='oracle':assert np.array_equal(b[k]['pdf'],c[k]['pdf'])
    for p in b.values():
        assert np.allclose(p['weights'].sum(axis=1),1,atol=2e-6)
        assert np.all(p['weights']>=0) and np.allclose(p['pdf'].sum(axis=1),1)

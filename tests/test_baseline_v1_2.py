import numpy as np
from photdis.baseline_v1_2 import (balanced_weights,conservative_threshold,
    mix_cached,main_component,single_mask)
from photdis.baseline_parallax import parallax_check


def test_class_balance():
    y=np.array([0]*100+[1]*3+[2]*17);w=balanced_weights(y)
    np.testing.assert_allclose([w[y==k].sum() for k in range(3)],40)


def test_inclusive_threshold_ties_and_budget():
    x=np.array([.9,.8,.8,.5,.1]*20)
    for budget in [0,.01,.02,.05,.2,.5,1]:
        th=conservative_threshold(x,budget)
        assert (x>=th).sum()<=int(budget*len(x))
    assert np.isinf(conservative_threshold([],0.01))


def test_main_branch_not_mixture_median_or_oracle():
    p=np.zeros((2,4,5));p[:,0,4]=1;p[:,1,0]=1;p[:,2,3]=1;p[:,3,2]=1
    raw=np.array([[.35,.4,.25],[.1,.8,.1]])
    mixed,w,lost=mix_cached(p,raw,np.array([[True]*3,[True,False,True]]))
    main,idx=main_component(p,w)
    assert idx.tolist()==[1,3]
    assert np.argmax(np.cumsum(mixed[0])>=.5)==3
    assert np.argmax(main[0])==0
    np.testing.assert_allclose(w.sum(1),1);assert lost[1]==.8
    assert single_mask(w,np.ones((2,3),bool),np.ones(2,bool),.7,.2).tolist()==[False,True]


def test_parallax_full_density_and_bin_partition():
    p=np.array([[.2,.5,.3],[.6,.1,.3]]);edges=np.linspace(0,3,4)
    g=np.array([16.,19.]);obs=np.array([-.1,.15]);err=np.array([.1,.05])
    full=parallax_check(p,edges,g,obs,err,np.full(2,-np.inf),np.full(2,np.inf))
    bins=[-np.inf,0,5,10,20,np.inf];prob=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        a=parallax_check(p,edges,g,obs,err,np.full(2,lo),np.full(2,hi));prob.append(a['selection_probability'])
        np.testing.assert_allclose(a['conditional_density']*a['selection_probability'],full['conditional_density'],rtol=1e-12)
    np.testing.assert_allclose(np.sum(prob,axis=0),1,atol=1e-14)
    assert np.all((full['pit']>=0)&(full['pit']<=1))

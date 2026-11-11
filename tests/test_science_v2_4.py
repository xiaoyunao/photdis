import numpy as np
from photdis.science_v2_4 import distance_bin_mass,cluster_mean_interval,conditional_prior,pdf_reference_shape


def test_joint_distance_mass_is_not_independent_product():
    # g=20: M=6 gives near dwarf, M=0 gives distant evolved star.
    edges=np.arange(-.5,7.);p=np.zeros((1,4,7));p[0,0,6]=.9;p[0,1,0]=.1
    m=distance_bin_mass(p,edges,np.array([20.]),[0,10,np.inf])
    np.testing.assert_allclose(m.sum(axis=2),p.sum(axis=2))
    assert m[0,1,0]==0 and p[0,1].sum()*m[0,:,0].sum()>.08
    np.testing.assert_allclose(m.sum(),1)


def test_prior_unseen_cells_and_cluster_repeats():
    y=np.array([0,0,1,2]);prior,p=conditional_prior(y,[0,0,1,1],[0,99])
    np.testing.assert_allclose(p[1],prior)
    np.testing.assert_allclose(p.sum(axis=1),1)
    a=cluster_mean_interval([1,1,3,3],[10,10,20,20])
    assert a['clusters']==2 and a['mean']==2 and a['low']==1 and a['high']==3
    b=cluster_mean_interval([2,2,2],[10,10,10]);assert b['low'] is None


def test_peak_reference_and_missing_reference():
    edges=np.arange(-2,12.01,.01);x=(edges[:-1]+edges[1:])/2
    p=np.exp(-.5*((x-5)/.15)**2)+.2*np.exp(-.5*((x-0)/.15)**2);p/=p.sum()
    a=pdf_reference_shape(np.tile(p,(2,1)),edges,[20,20],[20,np.nan])
    assert a['peak_near_reference'][0] and a['peak_count'][0]==2
    assert .15<a['mass_within1'][0]<.18 and np.isnan(a['mass_within1'][1])

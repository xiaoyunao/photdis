import numpy as np
from photdis.v3.population import SharedCounts
from photdis.v3_1.count_mixture import mix_counts


def test_shared_calibration_count_covariance_matches_single_joint_mixture():
    # Strong calibration changes create catalog-wide covariance; averaging
    # conditional covariances alone misses this independently known term.
    p=np.array([[[.1,.9],[.3,.7]],[[.2,.8],[.4,.6]],[[.8,.2],[.9,.1]],[[.9,.1],[.7,.3]]])
    state_weights=np.array([.25,.75]);within=[np.array([.3,.7]),np.array([.6,.4])];results=[]
    for j in range(2):
        acc=SharedCounts([10,11],2,within[j],seed=41+j);acc.add(p[2*j:2*j+2],[10,11]);results.append(acc.result())
    mixed=mix_counts(results,state_weights,[2,3])
    weight=np.concatenate([state_weights[j]*within[j] for j in range(2)])
    direct=SharedCounts(np.arange(4),2,weight);direct.add(p,np.arange(4));truth=direct.result()
    for key in ('mean','covariance','conditional_covariance','shared_covariance'):
        np.testing.assert_allclose(mixed[key],truth[key],atol=1e-14)
    assert mixed['between_state_covariance'][0,0]>.2
    np.testing.assert_allclose(mixed['covariance'].sum(axis=1),0,atol=1e-14)
    assert (mixed['count_draws'].sum(axis=1)==2).all()

import numpy as np
from photdis.v3_1.predictive import SharedPredictive,mix_predictive


def test_shared_predictive_is_not_product_of_marginal_predictives():
    # State0 favors the first object; state1 favors the second. The catalog
    # shares a state, so its density is 9, not the independent mixture's 25.
    response=np.array([[4.5,1.],[.5,9.]])
    entry=np.array([[.5,1.],[.5,1.]])
    ratio=np.log(response)-np.log(entry)
    value=SharedPredictive([.5,.5]);point=value.add(ratio,np.ones(2));result=value.result()
    np.testing.assert_allclose(np.exp(point),[5.,5.])
    np.testing.assert_allclose(np.exp(result['joint_log_density']),9.)
    assert result['joint_importance_ess']==2.
    # Selection normalization occurs within each population draw, before mixing.
    separate=SharedPredictive([.25,.75]);p=separate.add(np.log([[9.],[1.]]),[1.])
    np.testing.assert_allclose(np.exp(p),[3.])
    # Chunk boundaries do not create new independent global states.
    streamed=SharedPredictive([.5,.5]);streamed.add(ratio[:,:1],[1.]);streamed.add(ratio[:,1:],[1.])
    np.testing.assert_array_equal(streamed.result()['log_density_per_draw'],result['log_density_per_draw'])


def test_predictive_state_mixture_matches_direct_catalog_integral():
    densities=np.array([[2.,3.],[4.,1.],[.5,8.],[3.,2.]])
    results=[]
    for j,w in enumerate(([.2,.8],[.6,.4])):
        acc=SharedPredictive(w);acc.add(np.log(densities[2*j:2*j+2]),[.5,1.5]);r=acc.result();r['draw_weights']=np.array(w);results.append(r)
    value=mix_predictive(results,[.3,.7]);w=np.array([.06,.24,.42,.28])
    np.testing.assert_allclose(np.exp(value['joint_log_density']),w@densities.prod(axis=1))
    np.testing.assert_allclose(np.exp(value['generalized_design_joint_log_density']),w@(densities[:,0]**.5*densities[:,1]**1.5))
    assert not value['joint_supported']  # Four quadrature states cannot claim ESS100.

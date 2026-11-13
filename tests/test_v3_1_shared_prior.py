import numpy as np
from photdis.v3.inference import kernel_loglike
from photdis.v3.likelihood import normalize_joint
from photdis.v3_1.shared_prior import marginalize_native,geographic_fields
from photdis.v3_1.diagnostics import control_log_density
from photdis.v3.target import sed_tilt_features
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from test_v3_inference import model,observation


def test_shared_prior_matches_average_of_separately_normalized_posteriors():
    m,o=model(),observation();grid=np.unique(np.r_[np.linspace(-15,45,3001),DISTANCE_MU_BOUNDARIES])
    lk,_=kernel_loglike(m,o,grid);rng=np.random.default_rng(19)
    theta=rng.dirichlet(np.ones(15),size=6).reshape(6,3,5);beta=rng.normal(0,.5,(6,2));weights=np.array([.1,.05,.2,.3,.1,.25])
    result=marginalize_native(m,grid,lk,theta,beta,weights,batch=2)
    expected=np.zeros_like(result['posterior'].joint_density);features,_=sed_tilt_features(m)
    for th,be,w in zip(theta,beta,weights):
        p=normalize_joint(grid,control_log_density(m,o,grid,theta=th,component_tilt=features@be))
        expected+=w*p.joint_density
    np.testing.assert_allclose(result['posterior'].joint_density,expected,atol=1e-13)
    assert abs(result['log_normalization'])<1e-12 and result['joint_delta']<1e-12
    # A plug-in average prior is not the same estimand.
    plugin=normalize_joint(grid,control_log_density(m,o,grid,theta=np.einsum('s,scd->cd',weights,theta),component_tilt=features@(weights@beta)))
    assert abs(plugin.joint_density-expected).max()>1e-4


def test_geographic_ownership_never_confuses_cell_and_field_ids():
    result,existing=geographic_fields([355.,5.,15.,25.],[0.]*4,[0,1,2,3],{1:[350,-2,10,2],2:[0,-2,20,2]})
    np.testing.assert_array_equal(result,[1,1,2,10003])
    np.testing.assert_array_equal(existing,[True,True,True,False])

import numpy as np
from test_v3_inference import model, observation
from photdis.v3.inference import infer_reference, target_posterior
from photdis.v3.compression import compress
from photdis.v3.target import component_response, joint_coefficients
from photdis.v3.marginal import marginal_coefficients
from photdis.v3.marginal_cdf import cdf_product


def test_analytic_continuous_marginal_matches_native_likelihood_and_joint_bins():
    m=model(); o=observation(); p=infer_reference(m,o)['posterior']; compact=compress(p,2)
    theta=np.random.default_rng(91).dirichlet(np.ones(15),8).reshape(8,3,5)
    weights=np.arange(1,9)/36
    coefficient=joint_coefficients(m,theta,np.zeros((8,2)))
    mean,_=marginal_coefficients(component_response(compact,m)[None],coefficient,weights)
    got=cdf_product(compact,m,mean[0])
    native=[target_posterior(m,o,p.mu,t)[0] for t in theta]
    expected=sum(w*x.distance_bins().reshape(2,2,3,6).sum(axis=(0,1)) for w,x in zip(weights,native))
    np.testing.assert_allclose(got['class_distance'],expected,atol=5e-4)
    np.testing.assert_allclose(got['class_cdf'][:,-1],expected.sum(axis=1),atol=5e-4)
    assert np.diff(got['class_cdf'],axis=1).min() >= -1e-14
    assert abs(got['integral']-1)<5e-5
    finer=cdf_product(compact,m,mean[0],maximum_step=.0125)
    np.testing.assert_allclose(got['mu'],finer['mu'],atol=.0002)
    np.testing.assert_allclose(got['class_distance'],finer['class_distance'],atol=1e-12)

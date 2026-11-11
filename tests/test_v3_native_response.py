import numpy as np
from test_v3_inference import model,observation
from photdis.v3.inference import infer_reference,target_posterior
from photdis.v3.native_response import native_response
from photdis.v3.target import joint_coefficients,joint_draw_probabilities
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.batch_hierarchy import objective_batch


def test_unpruned_response_matches_native_forward_posterior():
    m=model();o=observation();p=infer_reference(m,o)['posterior']
    r=native_response(m,o,p.mu,p.log_evidence)
    theta=np.random.default_rng(3).dirichlet(np.ones(15),4).reshape(4,3,5)
    coef=joint_coefficients(m,theta,np.zeros((4,2)))
    got=joint_draw_probabilities(r[None],coef)[:,0]
    expected=np.array([target_posterior(m,o,p.mu,t)[0].distance_bins().reshape(2,2,3,6).sum(axis=(0,1)) for t in theta])
    np.testing.assert_allclose(got,expected,atol=4e-5)


def test_value_batch_matches_single_objective_across_multiple_fields():
    m=model();rng=np.random.default_rng(4)
    response=rng.uniform(.1,2,(20,2,5));selection=rng.uniform(.1,.9,(20,2,5))
    h=FastFieldPopulation(response,selection,m,np.arange(20)%3,rng.uniform(.1,2,20))
    z=rng.normal(size=(25,h.dimension))
    expected=np.array([h.objective(x)[0] for x in z])
    np.testing.assert_allclose(objective_batch(h,z,batch_size=7),expected,rtol=2e-14,atol=1e-12)

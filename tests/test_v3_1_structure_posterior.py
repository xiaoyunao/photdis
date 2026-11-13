import numpy as np
from photdis.v3_1.structure import structure_quadrature
from photdis.v3_1.structure_posterior import common_structure_posterior


def test_shared_structure_marginal_matches_one_star_bayes_identity():
    mu,lm,z,lz=structure_quadrature(16,128)
    rs=np.log(np.full((1,1,len(mu)),.8));rb=np.log([[.2]])
    result=common_structure_posterior(rs,rb,np.full(rs.shape,.25),[[.25]],[0],mu,lm,z,lz)
    np.testing.assert_allclose(result['log_evidence'],np.log(2),atol=1e-12)
    np.testing.assert_allclose(result['mean'],[.8,.2],atol=1e-12)
    np.testing.assert_allclose(result['covariance'],[[.16,-.16],[-.16,.16]],atol=1e-12)
    np.testing.assert_allclose(result['common_mu_probability'],np.exp(lm),atol=1e-12)


def test_response_prior_updates_once_for_the_whole_catalog():
    mu,lm,z,lz=structure_quadrature(16,64)
    # Each response is independent of membership, but contributes its own
    # likelihood for both stars: posterior response odds must be4, not2.
    rs=np.log(np.broadcast_to(np.array([.2,.4])[:,None,None],(2,2,len(mu))))
    rb=np.log(np.broadcast_to(np.array([.2,.4])[:,None],(2,2)))
    result=common_structure_posterior(rs,rb,np.full(rs.shape,.5),np.full((2,2),.5),[0,0],mu,lm,z,lz)
    np.testing.assert_allclose(result['response_weights'],[.2,.8],atol=1e-12)
    np.testing.assert_allclose(result['log_evidence'],np.log(.5*(.4**2+.8**2)),atol=1e-12)
    np.testing.assert_allclose(result['mean'],[1,1],atol=1e-12)
    assert result['shared_covariance'][0,0]>.5

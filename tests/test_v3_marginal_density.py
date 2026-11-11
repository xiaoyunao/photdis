import numpy as np
from scipy.special import logsumexp
from test_v3_inference import model,observation
from photdis.v3.inference import infer_reference,target_posterior
from photdis.v3.compression import compress
from photdis.v3.target import component_response,joint_coefficients
from photdis.v3.marginal import marginal_coefficients
from photdis.v3.marginal_density import native_marginal


def test_full_likelihood_reconstruction_preserves_tiny_log_tails():
    m=model();o=observation();p=infer_reference(m,o)['posterior'];record=compress(p,2)
    theta=np.random.default_rng(8).dirichlet(np.ones(15),4).reshape(4,3,5)
    co=joint_coefficients(m,theta,np.zeros((4,2)))
    average,_=marginal_coefficients(component_response(record,m)[None],co)
    got,normalization=native_marginal(m,o,p.mu,average[0],p.log_evidence)
    expected=[]
    for t in theta:
        q,_=target_posterior(m,o,p.mu,t)
        with np.errstate(divide='ignore'):expected.append(np.log(q.joint_density.sum(axis=0)))
    expected=logsumexp(expected,axis=0)-np.log(4)
    # The independent density-valued oracle underflows below ~745; compare in
    # its representable range while requiring our log-space tails stay finite.
    valid=np.isfinite(expected)&(expected> -700)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got[valid],expected[valid],atol=2e-4)
    assert np.any(got< -100) and abs(normalization)<.001

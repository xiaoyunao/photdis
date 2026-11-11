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


def test_native_class_joint_retains_all_kernels_and_normalizes_each_draw():
    from photdis.v3.marginal_density import native_joint_marginal
    from photdis.v3.native_response import native_response
    from photdis.v3.likelihood import ContinuousPosterior
    m=model();o=observation();p=infer_reference(m,o)['posterior']
    theta=np.random.default_rng(18).dirichlet(np.ones(15),8).reshape(8,3,5)
    from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
    grid=np.unique(np.r_[p.mu,DISTANCE_MU_BOUNDARIES])
    response=native_response(m,o,grid,p.log_evidence)
    co=joint_coefficients(m,theta,np.zeros((8,2)));mean,_=marginal_coefficients(response[None],co)
    logp,meta=native_joint_marginal(m,o,grid,mean[0],p.log_evidence)
    got=ContinuousPosterior(grid,np.exp(logp),0.,{}).distance_bins()
    from photdis.v3.target import joint_draw_probabilities
    expected=joint_draw_probabilities(response[None],co).mean(axis=0)[0]
    np.testing.assert_allclose(got,expected,atol=1e-12)
    assert abs(meta['log_normalization'])<1e-12 and 0<meta['binary_probability']<1

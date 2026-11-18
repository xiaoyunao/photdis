"""Scientific regressions for the new CMD and selection-conditioned controls."""
import numpy as np
from scipy.special import ndtr
from halo_filtering import envelope, accepted, channel_weights
from halo_noise import effective_errors
from halo_power import selected_components, simulate


def test_noise_declared_variance_matches_actual_injected_variance():
    base=np.array([[.03,.01,.02]])
    donor=np.array([[.01,.04,.02]])
    extra,effective=effective_errors(base,donor)
    np.testing.assert_allclose(effective**2,base**2+extra**2)
    np.testing.assert_allclose(effective,np.maximum(base,donor))
    assert extra[0,0]==0


def test_joint_preselection_branch_mass_has_correct_conditional_fraction():
    error=np.array([.2]);mu=np.array([[10.]])
    prepared=selected_components(mu,np.ones_like(mu),error,'gaia',np.array([[-np.inf,5.]]),.3,0)
    z0=ndtr(0.);z1=ndtr(5-10**(-.4)/.2)
    expected=.3*z1/(.7*z0+.3*z1)
    np.testing.assert_allclose(prepared[-1],expected)
    assert expected>.3
    # With no selection, the same preselection mass is retained exactly.
    free=selected_components(mu,np.ones_like(mu),error,'gaia',None,.3,0)
    np.testing.assert_allclose(free[-1],.3)
    _,valid=simulate(prepared,error,np.random.default_rng(7))
    assert valid.all()


def test_overlapping_cmd_nodes_are_not_exclusive_distance_memberships():
    curve=envelope(np.array([3.,8.]),np.array([.2,.7]))
    mag=np.array([[20.,19.6,19.4]])
    template={0:[curve],1:[curve]}
    error=dict(g_centers=np.array([18.,23.]),sigma_g=np.full((2,2),.02))
    result=accepted(mag,mag[:,0],np.array([0]),np.array([14.9,15.,15.1]),template,error)
    assert result.all()  # A single source legitimately enters several nodes.
    mu=np.round(np.arange(14,20.001,.1),1)
    w=channel_weights(mu,[(10**.8,10),(10,15),(15,26),(26,50),(50,100)])
    np.testing.assert_allclose(w.sum(axis=1),1.)
    assert (w>=0).all() and (w[0,mu>15.1]==0).all()

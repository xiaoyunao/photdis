"""Regressions for observed selection, shared-band PDFs and sky rematching."""
import numpy as np
from sv18_filter import observable, scan, recover
from sv18_populations import match
from sv18_predictive import gaussian_mixture
from paper_v2_reference import reference_uncertainty_mu
from sv18_observable import selected_fraction
from sv18_benchmark import fit, predict


def test_random_depth_selection_integrates_partial_magnitude_bins():
    result=selected_fraction(np.array([19.,20.,21.]),np.array([18.,19.,20.,21.,22.]))
    np.testing.assert_allclose(result,[1.,2/3,1/3,0.])
    result=selected_fraction(np.array([19.5]),np.array([19.,20.]))
    np.testing.assert_allclose(result,[.5])


def test_empirical_relation_supports_a_single_qualified_color_cell():
    mags=np.tile([18.45,18.,17.75],(24,1))
    training=dict(m=mags,y=np.linspace(4.,5.,24))
    model=fit(training)
    q,distance,count=predict(model,mags[:2],np.zeros((2,3)),np.array([1,2]),0.)
    assert q.shape==(2,5)
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=0).all()
    assert (count==24).all()


def test_geometric_one_sigma_and_95_halfwidth_have_distinct_units():
    parallax=np.array([1.])
    error=np.array([.02])
    t=dict(p_corr=parallax,p_err=error,
           gaia_lo=1/(parallax+1.96*error),gaia_hi=1/(parallax-1.96*error))
    sigma,half=reference_uncertainty_mu(t,'gaia',np.array([0]))
    np.testing.assert_allclose(sigma,5/np.log(10)*error/parallax)
    np.testing.assert_allclose(half/sigma,1.96,rtol=.001)


def test_observed_r_gate_includes_extinction():
    clear=observable(np.zeros(20),20).reshape(29,110)
    dusty=observable(np.full(20,.10),20).reshape(29,110)
    assert np.all(dusty<=clear+1e-12)
    assert dusty.sum()<clear.sum()
    assert clear[:,90:].sum()==0


def test_selected_template_recovers_true_grid_node(monkeypatch):
    import sv18_filter
    monkeypatch.setattr(sv18_filter.old,'MU',np.array([10.,11.,12.]))
    sig=np.array([[.8,.2,0],[.2,.8,0],[.1,.2,.7]])
    z,_,_,weight=scan(sig,1000*sig[1],np.zeros(3),0)
    index,lo,hi=recover(z,weight)
    assert index==1
    assert lo<=11.<=hi


def test_sky_bootstrap_rematch_respects_multiplicity_and_support():
    x=np.zeros((12,1))
    pop=np.repeat([1,2],6)
    delta=np.r_[np.zeros(6),np.ones(6)*.38]
    value,cells,balance=match(x,pop,delta)
    np.testing.assert_allclose(value,.38)
    assert len(cells)==1 and cells[0][1]==6 and balance.sum()==12
    weight=np.r_[np.zeros(3,int),np.ones(9,int)]
    value,cells,_=match(x,pop,delta,weight)
    assert np.isnan(value) and not cells


def test_gaussian_observation_cdf_conditioning_uses_signed_values():
    edges=np.tile([10.,10.001],(2,1))
    mass=np.ones((2,1))
    observed=np.array([-.03,.10])
    error=np.array([.1,.1])
    window=np.tile([-np.inf,5.],(2,1))
    pit,density=gaussian_mixture(edges,mass,observed,error,'gaia',window)
    assert np.isfinite(pit).all() and np.isfinite(density).all()
    assert 0<=pit[0]<pit[1]<=1

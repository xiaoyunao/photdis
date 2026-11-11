import numpy as np
from photdis.population_v2_5 import component_weights,selected_probabilities,posterior_grid,weighted_quantile


def test_selection_normalization_is_not_raw_mixture():
    response=np.array([[.4,.1],[.05,.05]])
    eta=np.array([[.5,.5]])
    np.testing.assert_allclose(selected_probabilities(eta,response),[[.75,.25]])
    # Explicit rejection sampler supplies an independent check of denominator.
    rng=np.random.default_rng(17);j=rng.integers(0,2,200000);u=rng.random(len(j))
    selected=u<response.sum(axis=1)[j];first=u<response[:,0][j]
    assert abs(first.sum()/selected.sum()-.75)<.005


def test_degenerate_response_preserves_prior_and_far_mass():
    base=np.array([.55,.1,.2,.15]);theta=np.zeros((1,4))
    w=component_weights(theta,base).reshape(4,5)
    np.testing.assert_allclose(w.sum(axis=1),base)
    assert w[:,3:].sum()>.19
    r=np.tile([.1,.2,.05],(20,1))
    t,s,p=posterior_grid([10,20,5],r,base,power=9)
    np.testing.assert_allclose(p,1/512,atol=1e-14)
    lo,_,hi=weighted_quantile(t[:,0],p[0]);assert lo< -2.7 and hi>2.7


def test_selection_changes_conditional_component_counts():
    base=np.ones(4)/4;r=np.eye(20)*np.linspace(.1,.9,20)[:,None]
    t,s,p=posterior_grid(np.ones(20),r,base,power=8)
    np.testing.assert_allclose(s.sum(axis=1),1)
    np.testing.assert_allclose(p.sum(axis=1),1)
    raw=component_weights(t,base)
    assert np.max(abs(raw-s))>.01

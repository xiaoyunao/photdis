import numpy as np
from photdis.v3_1.structure_draws import realized_structure_counts,count_draw_diagnostics


def test_realized_catalog_uses_one_global_state_and_preserves_integer_total():
    # Opposite essentially deterministic response states. Independent per-star
    # mixing would concentrate counts near20 and destroy the correct0/40 modes.
    rs=np.array([np.full((40,1),-1000.),np.full((40,1),1000.)])
    result=realized_structure_counts(rs,np.zeros((2,40)),np.zeros(40),[0.],np.array([[[.5]],[[.5]]]),draws=65536,seed=320601)
    counts=result['realized_counts']; assert counts.dtype==np.int64
    assert set(counts[:,0])=={0,40} and (counts.sum(axis=1)==40).all()
    np.testing.assert_array_equal(counts[:,0],40*result['response_index'])
    check=count_draw_diagnostics(counts,[20,20],[[400,-400],[-400,400]])
    assert check['passed'] and check['quantiles'][0][0]==0 and check['quantiles'][0][-1]==40


def test_conditional_bernoulli_counts_match_binomial_moments():
    p=.3; n=12; rs=np.full((1,n,1),np.log(p/(1-p)))
    counts=realized_structure_counts(rs,np.zeros((1,n)),np.zeros(n),[0.],np.ones((1,1,1)),draws=65536,seed=320601)['realized_counts']
    check=count_draw_diagnostics(counts,[n*p,n*(1-p)],n*p*(1-p)*np.array([[1,-1],[-1,1]]))
    assert check['passed'] and abs(check['empirical_member_variance']-n*p*(1-p))<.04

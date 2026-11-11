import numpy as np
from v3_0_transfer_plan import retain_responses


def test_response_compression_bounds_correlated_counts():
    w=np.array([.7,.3-5e-11,5e-11]);ix,new,delta=retain_responses(w,1e-10)
    # Each response deterministically puts all stars in one of three states:
    # this is maximally correlated, not independent-star uncertainty.
    x=np.array([[100.,0.],[0.,100.],[50.,50.]])
    original=w@x; compressed=new@x[ix]
    assert np.max(abs(original-compressed))<=100*delta+1e-12
    cov=(x-original).T@(w[:,None]*(x-original))
    cut=(x[ix]-compressed).T@(new[:,None]*(x[ix]-compressed))
    assert np.max(abs(cov-cut))<=3*100**2*delta+1e-10
    assert np.array_equal(ix,[0,1])


def test_uniform_responses_cannot_be_pruned():
    ix,w,delta=retain_responses(np.ones(32)/32,1e-10)
    assert len(ix)==32 and delta==0

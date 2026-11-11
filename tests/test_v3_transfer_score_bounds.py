import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from v3_0_transfer_evaluate import gaussian_score_bounds


def test_gaussian_observation_mixture_bounds_cover_adversarial_omitted_response():
    # The omitted response can place all its latent mass exactly at the observation.
    retained=norm.logpdf(8.,loc=0.,scale=.3);delta=.002
    lo,hi=gaussian_score_bounds(retained,.3,np.log(delta))
    actual=logsumexp([np.log1p(-delta)+retained,np.log(delta)+norm.logpdf(8.,loc=8.,scale=.3)])
    assert lo<=actual<=hi+1e-12
    np.testing.assert_allclose(actual,hi,atol=1e-12)
    # exp(-1000) cannot safely be discarded from a conflicting observation score.
    lo,hi=gaussian_score_bounds(-2000.,.3,-1000.)
    assert hi>-1001 and lo==-2000

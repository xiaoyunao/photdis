import numpy as np
from scipy.stats import multivariate_normal
from photdis.v3.structured_evidence import structured_bridge,structured_covariance


def test_high_dimension_bridge_recovers_known_normalizer():
    rng=np.random.default_rng(51);d=58;g=4;b=6
    a=np.eye(d);a[g:,:g]=rng.normal(0,.2,(d-g,g));cov=a@a.T
    samples=rng.multivariate_normal(np.zeros(d),cov,size=4000)
    mean,estimated=structured_covariance(samples,g,b)
    assert np.linalg.eigvalsh(estimated).min()>0
    target=multivariate_normal(np.zeros(d),cov)
    got=structured_bridge(lambda x:target.logpdf(x)+37.2,samples,g,b,proposal_draws=4096,seed=7)
    assert got['fixed_point_converged'] and got['proposal_bridge_ess']>500
    assert abs(got['log_integral']-37.2)<.06
    assert got['half_log_integral_difference']<.1

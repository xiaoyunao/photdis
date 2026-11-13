import numpy as np
from photdis.v3_1.finite_counts import finite_count_marginals
from photdis.v3.population import SharedCounts


def test_finite_shared_marginals_preserve_correlation_and_covariance():
    p=np.array([[[.1,.9],[.1,.9]],[[.9,.1],[.9,.1]]]);w=np.array([.5,.5])
    mass=finite_count_marginals(p,w)
    np.testing.assert_allclose(mass,[[.41,.18,.41],[.41,.18,.41]])
    # Mixing each star first would incorrectly give [.25,.5,.25].
    accumulator=SharedCounts([0,1],2,w);accumulator.add(p,[0,1]);counts=accumulator.result()
    support=np.arange(3);mean=mass@support;variance=mass@(support**2)-mean**2
    np.testing.assert_allclose(mean,counts['mean'])
    np.testing.assert_allclose(variance,np.diag(counts['covariance']))
    np.testing.assert_allclose(counts['covariance'].sum(axis=1),0,atol=1e-15)

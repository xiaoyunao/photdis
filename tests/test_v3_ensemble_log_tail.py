import numpy as np
from v3_0_ensemble import mixture_log_density


def test_tiny_response_weight_survives_conflicting_observation_tail():
    # A negligible response can still dominate density in a far posterior tail.
    result=mixture_log_density(np.array([[-2000.,0.],[0.,-2000.]]),np.array([0.,-1000.]))
    np.testing.assert_allclose(result,[-1000.,0.],atol=1e-12)

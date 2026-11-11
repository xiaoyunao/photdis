import numpy as np
from photdis.v3.structured_evidence import structured_bridge
from v3_0_parallel_bridge import parallel_bridge


def test_parallel_evaluation_exactly_preserves_bridge_proposals_and_result():
    rng=np.random.default_rng(9);x=rng.normal(size=(512,8))
    def logdensity(z):return -.5*(z@z)-4*np.log(2*np.pi)
    serial=structured_bridge(logdensity,x,4,2,proposal_draws=1024,seed=71)
    parallel=parallel_bridge(logdensity,x,4,2,proposal_draws=1024,seed=71,workers=2)
    for key in serial:np.testing.assert_equal(parallel[key],serial[key])

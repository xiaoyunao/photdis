import numpy as np
from v3_0_ensemble import mixture_log_density


def test_tiny_response_weight_survives_conflicting_observation_tail():
    # A negligible response can still dominate density in a far posterior tail.
    result=mixture_log_density(np.array([[-2000.,0.],[0.,-2000.]]),np.array([0.,-1000.]))
    np.testing.assert_allclose(result,[-1000.,0.],atol=1e-12)


def test_cdf_mixture_preserves_narrow_source_breakpoints():
    from v3_0_ensemble import mixture_cdf_grid
    # Separate, sub-grid branches: direct interpolation of each input is the
    # independent reference, not a second implementation of grid construction.
    x1=np.array([0., .503, .504, 1.])
    x2=np.array([0., .701, .702, 1.])
    y1=np.array([[0.,0.,.7,.7],[0.,0.,.3,.3]])
    y2=np.array([[0.,0.,.2,.2],[0.,0.,.8,.8]])
    curves=[(x1,y1),(x2,y2)];weights=[.3,.7]
    grid=mixture_cdf_grid(curves,boundaries=[.5,.75])
    expected_x=np.unique(np.r_[np.linspace(0,1,10001),x1,x2])
    def mixture(x):
        return sum(w*np.array([np.interp(x,knot,row) for row in cdf])
                   for w,(knot,cdf) in zip(weights,curves))
    combined=mixture(grid)
    reconstructed=np.array([np.interp(expected_x,grid,row) for row in combined])
    np.testing.assert_allclose(reconstructed,mixture(expected_x),rtol=0,atol=2e-15)
    coarse=np.linspace(0,1,101)
    lost=np.array([np.interp(expected_x,coarse,row) for row in mixture(coarse)])
    assert np.max(abs(lost-mixture(expected_x)))>.01
    np.testing.assert_allclose(combined[:,-1].sum(),1.)

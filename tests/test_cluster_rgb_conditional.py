"""Forward RGB error propagation and preregistered support checks."""
import numpy as np

from photdis.cluster_rgb_conditional import (grid_from_bins, posterior_on_grid,
                                             quantiles,
                                             residual_measurement_cov,
                                             support_bins)


def test_shared_band_covariance_matches_linear_transformation():
    sg, sr, sz = .03, .04, .05
    slope = np.array([[.3, -.2]])
    result = residual_measurement_cov(sg, sr, sz, slope, 2)[0]
    transform = np.array([[1-.3, -1, 0], [.2, 1, -1]])
    expected = transform @ np.diag([sg*sg, sr*sr, sz*sz]) @ transform.T
    np.testing.assert_allclose(result, expected)
    assert result[0,1] < 0


def test_support_requires_same_stage_two_cluster_overlap():
    data = {("m53","upper_rgb"):np.array([-.3]*5),
            ("ngc7006","upper_rgb"):np.array([-.2]*5),
            ("m53","lower_rgb"):np.array([1.2]*6),
            ("m3","lower_rgb"):np.array([2.2]*6)}
    bins = support_bins(data,[-.5,0,1,2,3],5)
    assert bins["upper_rgb"] == [[-.5,0.,5,5]]
    assert bins["lower_rgb"] == []
    assert len(grid_from_bins(bins["upper_rgb"],.1)) == 5


def test_unknown_abundance_prior_changes_posterior_without_training_histogram():
    model = {"beta":[[.7,.1,0,.15]],"intrinsic":[[.02**2]]}
    grid = np.linspace(0,2,101)
    p1 = posterior_on_grid([model],grid,[-1.46,-2.0],[.8,.2],
                           [.75],.01,.01,.01)
    p2 = posterior_on_grid([model],grid,[-1.46,-2.0],[.2,.8],
                           [.75],.01,.01,.01)
    np.testing.assert_allclose(p1.sum(),1)
    np.testing.assert_allclose(p2.sum(),1)
    assert abs(np.sum(grid*p1)-np.sum(grid*p2)) > .2


def test_posterior_quantile_stays_on_disconnected_supported_grid():
    grid = np.array([0.,.1,1.,1.1])
    p = np.array([.1,.39,.41,.1])
    assert quantiles(grid,p,(.5,))[0] == 1.

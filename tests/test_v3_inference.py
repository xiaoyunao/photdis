from types import SimpleNamespace
import numpy as np
from scipy.stats import norm
from photdis.v3.inference import infer_reference, basis_response, target_posterior, distance_basis
from photdis.v3.likelihood import flux_loglikelihood, trapezoid_weights


def model():
    m = np.array([[100., 120., 130., 150., 160., 170., -.5],
                  [10000., 12000., 13000., 15000., 16000., 17000., -1.5]])
    cov = np.array([np.diag(np.r_[(row[:6]*.05)**2, .1]) for row in m])
    return SimpleNamespace(weights=np.array([.8, .2]), means=m, covariances=cov,
        classes=np.array([[.98, .01, .01], [.01, .98, .01]]), system='DECam_DR11',
        binary_reference_fraction=.1, binary_brightness_factor=2.,
        negative_intrinsic_flux_probability=np.zeros((2, 6)))


def observation():
    return dict(flux=np.array([1., 1.2, np.nan, 1.5, -.1, np.nan]),
                flux_error=np.array([.05, .05, np.nan, .05, 1., np.nan]),
                band_mask=np.array([1, 1, 0, 1, 1, 0], bool),
                transmission=np.ones(6), photometric_system='DECam_DR11')


def test_same_color_luminosity_branches_survive_continuous_inference():
    p = infer_reference(model(), observation(), initial_step=.15)
    assert p['converged'] and abs(p['class_distance'].sum()-1) < 1e-12
    assert p['class_distance'][1, 4:].sum() > .05
    assert p['class_distance'][0, :3].sum() > .1
    assert len(p['modes']) >= 2


def test_population_basis_matches_explicit_joint_integration():
    m = model(); o = observation(); grid = np.linspace(0, 30, 601)
    logs, prior = basis_response(m, o, grid)
    theta = np.arange(1, 16, dtype=float).reshape(3, 5); theta /= theta.sum()
    p, provenance = target_posterior(m, o, grid, theta)
    expected_evidence = np.sum(theta*np.exp(logs))
    np.testing.assert_allclose(np.exp(p.log_evidence), expected_evidence, rtol=1e-10)
    assert p.distance_bins().sum() > .9999999
    # Full joint prior is reconstructable, not just a one-dimensional d prior.
    np.testing.assert_allclose(provenance['component_given_class'].sum(axis=0), 1)

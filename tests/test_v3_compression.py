import numpy as np
import pytest
from test_v3_inference import model, observation
from photdis.v3.inference import infer_reference, target_posterior
from photdis.v3.compression import compress, reweight


def test_joint_compression_retains_far_branch_and_population_reweighting():
    m = model(); o = observation(); result = infer_reference(m, o)
    p = result['posterior']; compact = compress(p, 2)
    assert compact['weighted_cdf_bound'] < .00021
    assert compact['omitted_mass'] < 1.01e-7
    baseline = reweight(compact, m, reference_prior=True)
    np.testing.assert_allclose(baseline['class_distance'], result['class_distance'], atol=3e-4)
    theta = np.arange(1, 16, dtype=float).reshape(3, 5); theta /= theta.sum()
    full, _ = target_posterior(m, o, p.mu, theta)
    expected = full.distance_bins().reshape(2, 2, 3, 6).sum(axis=(0, 1))
    changed = reweight(compact, m, theta=theta)
    np.testing.assert_allclose(changed['class_distance'], expected, atol=5e-4)
    np.testing.assert_allclose(changed['mu_quantiles'], full.quantiles(), atol=.012)
    assert changed['class_distance'][1, 4:].sum() > .05


def test_shared_luminosity_shift_is_exact_native_likelihood_translation():
    m = model(); o = observation(); p = infer_reference(m, o)['posterior']
    compact = compress(p, 2)
    shift = np.array([.15, -.12])
    direct = infer_reference(m, o, luminosity_shift=shift)
    shifted = reweight(compact, m, luminosity_shift=shift, reference_prior=True)
    np.testing.assert_allclose(shifted['class_distance'], direct['class_distance'], atol=5e-4)
    np.testing.assert_allclose(shifted['mu_quantiles'], direct['mu_quantiles'], atol=.012)
    invalid = dict(compact, omitted_mass=.01)
    with pytest.raises(ValueError, match='Omitted support'):
        reweight(invalid, m, reference_prior=True)

import numpy as np

from photdis.metal_poor_forward import (effective_mag_variance,
                                        expected_parallax, log_probability)


def sample():
    return {"gr0": np.array([1.0]), "rz0": np.array([0.5]),
            "g0": np.array([15.0]), "sigma_g": np.array([0.01]),
            "sigma_r": np.array([0.02]), "sigma_z": np.array([0.03]),
            "feh_cal": np.array([-2.0]), "desi_feh_err": np.array([0.1]),
            "gaia_parallax_error": np.array([0.1]), "p_corr": np.array([-0.1])}


def test_forward_likelihood_accepts_negative_parallax():
    data = sample()
    params = np.array([6.0, 1.0, 0.5, 0.0, -0.4, 0.2])
    assert np.isfinite(log_probability(params, data, True)).all()
    assert expected_parallax(params, data, True)[0] > 0


def test_shared_band_variance_and_iron_term():
    data = sample()
    params = np.array([6.0, 1.0, 0.5, 0.0, -0.4, 0.2])
    value = effective_mag_variance(params, data, True, 0.1)[0]
    expected = (0 * .01)**2 + (-.5 * .02)**2 + (.5 * .03)**2
    expected += .4**2 * (.1**2 + .1**2) + .2**2
    assert np.isclose(value, expected)

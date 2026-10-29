import numpy as np
from scipy.special import logsumexp
import sys
from pathlib import Path
import fitsio

from photdis.metal_poor_forward import (effective_mag_variance,
                                        expected_parallax, log_probability,
                                        LOG_PAR)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import metal_poor_forward_experiment as experiment


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


def test_posterior_centered_quadrature_resolves_narrow_gaia_likelihood():
    data = sample()
    data["g0"][0] = 16
    data["gaia_parallax_error"][0] = .01
    params = np.array([6.0, 2.0, .5, 0.0, .3])
    p = expected_parallax(params, data, False)[0]
    a = LOG_PAR * np.sqrt(effective_mag_variance(params, data, False, .1)[0])
    latent = np.linspace(-30, 30, 300001)
    step = latent[1] - latent[0]
    for observed in (-.1, .2, 1.0, 2.0):
        data["p_corr"][0] = observed
        log_integrand = (-.5 * latent**2
                         -.5 * ((observed - p * np.exp(a * latent)) / .01)**2
                         - np.log(2 * np.pi * .01))
        direct = logsumexp(log_integrand) + np.log(step)
        computed = log_probability(params, data, False)[0]
        assert abs(computed - direct) < .001


def test_prepare_handles_fits_unicode_roles_and_unusable_layers(tmp_path, monkeypatch):
    dtype = [("source_id", "i8"), ("ls_id", "i8"), ("field", "S16"),
             ("role", "S24"), ("trusted_blue_core", "?"),
             ("parallax_layer", "S16"), ("gr0", "f4"), ("rz0", "f4"),
             ("g0", "f4"), ("sigma_g", "f4"), ("sigma_r", "f4"),
             ("sigma_z", "f4"), ("feh_cal", "f4"),
             ("desi_feh_err", "f4"), ("p_corr", "f4"),
             ("gaia_parallax_error", "f4"), ("desi_snr", "f4")]
    rows = np.zeros(4, dtype=dtype)
    rows["source_id"] = [1, 2, 1, 3]
    rows["ls_id"] = [11, 22, 11, 33]
    rows["role"] = [b"development", b"development",
                    b"reserved_validation", b"reserved_validation"]
    rows["parallax_layer"] = [b"high_ge20", b"negative", b"weak_0_10", b"unusable"]
    rows["trusted_blue_core"] = True
    for name, value in (("gr0", 1), ("rz0", .5), ("g0", 16),
                        ("sigma_g", .01), ("sigma_r", .01), ("sigma_z", .01),
                        ("feh_cal", -1), ("desi_feh_err", .1),
                        ("p_corr", .2), ("gaia_parallax_error", .1),
                        ("desi_snr", 20)):
        rows[name] = value
    paths = [tmp_path / "dev.fits", tmp_path / "reserved.fits"]
    fitsio.write(str(paths[0]), rows[:2])
    fitsio.write(str(paths[1]), rows[2:])
    monkeypatch.setattr(experiment, "JOINT", paths)
    selected, counts = experiment.prepare()
    assert counts == {"input_usable": 3, "identity_duplicates_removed": 1}
    assert selected["source_id"].tolist() == [1, 2]
    assert np.char.strip(selected["role"].astype("U")).tolist() == [
        "reserved_validation", "development"]

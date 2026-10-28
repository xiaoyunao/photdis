"""Check the five-band foreground-fraction perturbation algebra."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cold_v4_dust_sensitivity import COEFF, shift_dereddening


def test_half_foreground_keeps_observed_flux_and_changes_all_colors():
    original = {name: np.array([value], float) for name, value in
                {"g0": 18., "r0": 17., "gr0": 1., "rz0": .6,
                 "zw10": 1.2, "w1w20": .1}.items()}
    ag = np.array([.08])
    shifted = shift_dereddening(original, ag, .5)
    delta = {key: .5 * .08 * value / COEFF["g"]
             for key, value in COEFF.items()}
    assert np.isclose(shifted["g0"][0], 18 + delta["g"])
    assert np.isclose(shifted["r0"][0], 17 + delta["r"])
    for color, first, second in (("gr0", "g", "r"), ("rz0", "r", "z"),
                                 ("zw10", "z", "w1"), ("w1w20", "w1", "w2")):
        assert np.isclose(shifted[color][0] - original[color][0],
                          delta[first] - delta[second])
    assert original["g0"][0] == 18.
    assert np.array_equal(shift_dereddening(original, ag, 1)["g0"], original["g0"])

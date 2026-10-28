"""Optical replacement must propagate into g0 and every affected color."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cold_v4_ps1_relative_audit import substitute_grz


def test_ps1_substitution_preserves_reddening_and_updates_color_chain():
    ag, ar, az = .08, .05, .03
    table = {"g0": np.array([18.0, 17.0]),
             "gr0": np.array([1.1, 1.2]),
             "rz0": np.array([.5, .6]),
             "zw10": np.array([1.4, 1.5]),
             "mw_transmission_g": 10 ** (-np.array([ag, ag]) / 2.5),
             "mw_transmission_r": 10 ** (-np.array([ar, ar]) / 2.5),
             "mw_transmission_z": 10 ** (-np.array([az, az]) / 2.5)}
    # The first source changes by +0.02, -0.01 and +0.03 mag in g/r/z.
    ps1 = {"g": np.array([18 + ag + .02, 0]),
           "r": np.array([18 - 1.1 + ar - .01, 0]),
           "z": np.array([18 - 1.1 - .5 + az + .03, 0])}
    out, legacy_minus_ps1 = substitute_grz(table, np.array([True, False]), ps1)
    assert np.isclose(out["g0"][0], 18.02)
    assert np.isclose(out["r0"][0], 16.89)
    assert np.isclose(out["gr0"][0], 1.13)
    assert np.isclose(out["rz0"][0], .46)
    assert np.isclose(out["zw10"][0], 1.43)
    assert np.isclose(legacy_minus_ps1["g"][0], -.02)
    for name in ("g0", "gr0", "rz0", "zw10"):
        assert np.isclose(out[name][1], table[name][1])
    assert np.isclose(out["r0"][1], table["g0"][1] - table["gr0"][1])

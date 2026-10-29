import importlib.util
from pathlib import Path
import sys

import numpy as np


def test_single_band_substitution_changes_all_shared_colors():
    path = Path(__file__).resolve().parents[1] / "scripts/cold_v4_optical_response.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("optical_response", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = {"g": np.array([17.]), "r": np.array([16.]), "z": np.array([15.]),
             "g0": np.array([16.9]), "gr0": np.array([1.]),
             "rz0": np.array([1.]), "zw10": np.array([2.])}
    transformed = {"g": np.array([17.01]), "r": np.array([16.02]),
                   "z": np.array([15.03])}
    one = module.band_substitution(table, np.array([True]), transformed, "r")
    np.testing.assert_allclose(one["g0"], table["g0"])
    np.testing.assert_allclose(one["gr0"], [.98])
    np.testing.assert_allclose(one["rz0"], [1.02])
    np.testing.assert_allclose(one["zw10"], table["zw10"])

import importlib.util
from pathlib import Path
import sys

import numpy as np


def test_complete_input_match_decomposes_label_and_model():
    path = Path(__file__).resolve().parents[1] / "scripts/cold_v4_same_star_tiers.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("same_star_tiers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    south = {"level": np.array([1]), "gr0": np.array([1.0]),
             "rz0": np.array([.5]), "sigma_g": np.array([.01]),
             "sigma_r": np.array([.01]), "sigma_z": np.array([.01]),
             "mg_gaia": np.array([7.]), "mg_model": np.array([7.2]),
             "p_corr": np.array([2.]), "r0": np.array([17.]),
             "g0": np.array([18.])}
    north = {**south, "gr0": np.array([1.005]),
             "mg_gaia": np.array([8.]), "mg_model": np.array([7.7])}
    result = module.match_complete(south, north, "A_grz")
    assert result["n"] == 1
    assert abs(result["north_minus_south_residual"]["median"] - .5) < 1e-12
    assert abs(result["north_minus_south_gaia_mg"]["median"] - 1) < 1e-12

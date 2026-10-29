import importlib.util
from pathlib import Path
import sys

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/cold_v5_xd_compare.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("xd_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_band_perturbation_updates_direct_modulus_and_both_colors():
    mod = _module()
    table = {"g0": np.array([18.]), "gr0": np.array([1.2]),
             "rz0": np.array([.7]), "zw10": np.array([.4]),
             "w1w20": np.array([.1])}
    use = np.array([True])
    g, c = mod.perturbed_features(table, "C_grzW1W2", use, "g", .01)
    assert np.allclose(g, [18.01])
    assert np.allclose(c, [[1.21, .7, .4, .1]])
    r_g, r_c = mod.perturbed_features(table, "C_grzW1W2", use, "r", .01)
    assert np.allclose(r_g, [18.])
    assert np.allclose(r_c, [[1.19, .71, .4, .1]])


def test_interval_statistics_distinguishes_tail_and_coverage():
    mod = _module()
    truth = np.array([10., 12.])
    predicted = np.array([[9., 9.5, 10., 10.5, 11.],
                          [10., 10.5, 11., 11.5, 12.]])
    result = mod.statistics(truth, predicted)
    assert result["center_mag"] == -.5
    assert result["coverage95"] == 1.
    assert result["coverage68"] == .5
    assert result["abs_gt_0p5"] == .5


def test_empty_color_group_has_explicit_response_count():
    mod = _module()
    assert mod.response_stats(np.array([])) == {"n": 0}

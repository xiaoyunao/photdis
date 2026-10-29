import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/cold_v4_crossover_common.py"
    spec = importlib.util.spec_from_file_location("crossover_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_difference_of_medians_is_not_median_of_differences():
    mod = _module()
    a = np.array([0., 10., 11.])
    b = np.array([8., 10., 20.])
    result = mod.paired_centers(a, b, 10, 100)
    assert result["difference_of_medians"] == 0
    assert result["median_of_paired_differences"] == 8
    assert result["environment_0"]["nmad"] != result["environment_1"]["nmad"]
    assert len(result["difference_of_medians_bootstrap95"]) == 2

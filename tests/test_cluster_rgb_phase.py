"""Check that broad phase screens use Gaia CMD, never Legacy residuals."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cluster_rgb_phase", ROOT / "scripts/cluster_rgb_phase.py")
phase = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase)


def test_m3_phase_categories_are_mutually_exclusive():
    config = json.loads((ROOT / "configs/cluster_rgb_phase_regions_v1.json").read_text())["systems"]["m3"]
    labels = phase.classify(np.array([-1, .8, .8, -1, 5, np.nan]),
                            np.array([1.2, 1.1, .4, .85, .7, 1]), config)
    assert labels.tolist() == ["upper_rgb", "lower_rgb", "hb", "possible_agb",
                               "main_sequence", "invalid_gaia_cmd"]


def test_locus_uses_all_candidates_in_bin():
    result = phase.locus(np.array([0.1, 0.2, 0.3, 0.4, 0.45]),
                         np.array([.8, .9, 1.0, 1.1, 1.2]))
    assert len(result) == 1
    assert result[0]["n"] == 5
    assert result[0]["color_median"] == 1.0

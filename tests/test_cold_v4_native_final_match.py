"""Focused invariants for the final-field diagnostic matching."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cold_v4_native_final_match import match_rows


def rows(gr, residual, parallax):
    count = len(gr)
    return {"gr0": np.array(gr, float), "rz0": np.full(count, .6),
            "ag": np.full(count, .05), "p_corr": np.array(parallax, float),
            "bp_rp": np.full(count, 2.), "r0": np.full(count, 17.),
            "residual": np.array(residual, float),
            "ra": np.linspace(10., 12., count),
            "dec": np.linspace(20., 22., count)}


def test_match_is_one_to_one_and_parallax_subset_is_fixed():
    south = rows([1.0, 1.8], [0.0, -.2], [2.5, 1.5])
    north = rows([1.01, 1.02, 1.81], [.1, .9, -.1], [2.6, 2.7, 1.6])
    feature = (("gr0", .2), ("rz0", .15), ("ag", .02))
    all_pairs = match_rows(south, north, feature, "all", 12)
    near_pairs = match_rows(south, north, feature, "parallax_ge_2_mas", 12)
    assert all_pairs["n"] == 2
    assert np.isclose(all_pairs["north_minus_south_median"], .1)
    assert near_pairs["n"] == 1
    assert np.isclose(near_pairs["north_minus_south_median"], .1)

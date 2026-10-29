"""Compact frozen-v4 anchor sidecars carry selection, not an EBV column."""
from __future__ import annotations

import numpy as np

import cold_v5_blue_inventory as inventory


def test_training_blue_coverage_without_ebv(monkeypatch):
    table = {"grz_base": np.array([True, True]),
             "gr0": np.array([.8, 1.1]),
             "grz_clean": np.array([True, False]),
             "r": np.array([18., 19.])}
    monkeypatch.setattr(inventory, "load_training", lambda _: [
        (table, np.array([True, True]), None, {"name": "field"})])
    result = inventory.training_blue_coverage(None, .909)["field"]
    assert result["blue_below_frozen_gr_lower_edge"] == 1
    assert result["blue_clean_low_dust"] == 1

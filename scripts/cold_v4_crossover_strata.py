#!/usr/bin/env python3
"""Summarize paired crossover residuals by measured quality and source color."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from photdis.stage1 import require_compute, write_json

BASE = ROOT / "data/cold_v4_crossover"
RESULT = BASE / "strata.json"


def summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    center = float(np.median(values))
    return {"n": len(values), "median": center,
            "nmad": float(1.4826 * np.median(np.abs(values - center)))}


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    manifest = json.loads((BASE / "diagnostic.json").read_text())
    rows = np.load(BASE / "rows.npz")
    result = {"prepared_sha256": manifest["prepared_sha256"],
              "method": "paired-source diagnostic strata, no model fitting",
              "cells": {}}
    for key in manifest["cells"]:
        vector = lambda name: rows[f"{key}__{name}"]
        paired = vector("paired").astype(bool)
        sigma = vector("sigma_used")
        support = vector("support_ratio")
        color = vector("native_rz")
        ag = vector("source_ext_g")
        groups = {
            "all_paired": paired,
            "sigma_le_0p10": paired & (sigma <= .10),
            "sigma_gt_0p10": paired & (sigma > .10),
            "support_le_0p5": paired & (support <= .5),
            "support_gt_0p5": paired & (support > .5),
            "native_rz_lt_0p8": paired & (color < .8),
            "native_rz_0p8_to_1p4": paired & (color >= .8) & (color < 1.4),
            "native_rz_ge_1p4": paired & (color >= 1.4),
            "source_ag_le_0p05": paired & (ag <= .05),
            "source_ag_gt_0p05": paired & (ag > .05),
        }
        result["cells"][key] = {
            name: {metric: summary(vector(metric)[use]) for metric in (
                "native_residual", "degradation_increment", "degraded_total")}
            for name, use in groups.items()}
    write_json(RESULT, result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare native field residuals after outcome-blind color/dust matching."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.optimize import linear_sum_assignment

from photdis.stage1 import require_compute, write_json

BASE = ROOT / "data/cold_v4_crossover"
OUT = BASE / "native_match.json"


def source_vectors(rows, field):
    keys = [f"{field}|{env}|{depth}" for env in (0, 1) for depth in range(4)]
    residuals = np.stack([rows[f"{key}__native_residual"] for key in keys])
    first = np.argmax(np.isfinite(residuals), axis=0)
    residual = residuals[first, np.arange(residuals.shape[1])]
    values = np.stack([rows[f"{keys[0]}__{name}"] for name in (
        "native_gr", "native_rz", "source_ext_g")], axis=1)
    usable = np.isfinite(residual) & np.isfinite(values).all(axis=1)
    return values[usable], residual[usable]


def match(a, b, resid_a, resid_b, scales, max_radius, seed):
    delta = (a[:, None, :len(scales)] - b[None, :, :len(scales)]) / scales
    distances = np.sqrt(np.sum(delta**2, axis=2))
    ia, ib = linear_sum_assignment(distances)
    keep = distances[ia, ib] <= max_radius
    ia, ib = ia[keep], ib[keep]
    diff = resid_b[ib] - resid_a[ia]
    if not len(diff):
        return {"matched_pairs": 0}
    rng = np.random.default_rng(seed)
    boot = [np.median(diff[rng.integers(len(diff), size=len(diff))])
            for _ in range(1000)]
    return {"matched_pairs": len(diff), "median_final_minus_cal_residual":
            float(np.median(diff)), "median_bootstrap95":
            np.quantile(boot, [.025, .975]).tolist(),
            "median_matching_radius": float(np.median(distances[ia, ib])),
            "p95_matching_radius": float(np.quantile(distances[ia, ib], .95)),
            "median_absolute_feature_difference": {
                name: float(np.median(abs(a[ia, index] - b[ib, index])))
                for index, name in enumerate(("g-r", "r-z", "A_g"))},
            "residual_nmad": float(1.4826 * np.median(abs(diff - np.median(diff))))}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = np.load(BASE / "rows.npz")
    a, ra = source_vectors(rows, 0)
    b, rb = source_vectors(rows, 1)
    result = {"method": "one-to-one minimum-cost native-source matching, no residual used in matching",
              "source_counts": [len(a), len(b)],
              "unmatched_native_medians": [float(np.median(ra)), float(np.median(rb))],
              "color_only": match(a, b, ra, rb, np.array([.2, .15]), 1.5, 10),
              "color_and_extinction": match(a, b, ra, rb,
                                              np.array([.2, .15, .02]), 1.5, 11)}
    write_json(OUT, result)


if __name__ == "__main__":
    main()

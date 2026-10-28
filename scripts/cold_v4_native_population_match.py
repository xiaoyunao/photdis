#!/usr/bin/env python3
"""Match frozen crossover sources on Gaia labels as a diagnostic only."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.optimize import linear_sum_assignment

from photdis.stage1 import require_compute, write_json

BASE = ROOT / "data/cold_v4_crossover"
OUT = BASE / "native_population_match.json"
FEATURE_SETS = {
    "legacy_colors_ag": (("native_gr", .2), ("native_rz", .15), ("source_ag", .02)),
    "plus_parallax": (("native_gr", .2), ("native_rz", .15), ("source_ag", .02),
                      ("p_corr", 1.0)),
    "plus_gaia_color": (("native_gr", .2), ("native_rz", .15), ("source_ag", .02),
                        ("gaia_bp_rp", .4)),
    "plus_parallax_gaia_color": (("native_gr", .2), ("native_rz", .15),
                                  ("source_ag", .02), ("p_corr", 1.0),
                                  ("gaia_bp_rp", .4)),
    "plus_gaia_g": (("native_gr", .2), ("native_rz", .15), ("source_ag", .02),
                    ("gaia_phot_g_mean_mag", 1.0)),
}


def run_match(rows, features, max_radius, seed):
    arrays = []
    residuals = []
    for field in (0, 1):
        matrix = np.column_stack([rows[f"{field}__{name}"] / scale
                                  for name, scale in features])
        residual = rows[f"{field}__native_residual"]
        valid = np.isfinite(matrix).all(axis=1) & np.isfinite(residual)
        arrays.append(matrix[valid])
        residuals.append(residual[valid])
    distance = np.sqrt(np.sum((arrays[0][:, None] - arrays[1][None])**2, axis=2))
    i, j = linear_sum_assignment(distance)
    keep = distance[i, j] <= max_radius
    i, j = i[keep], j[keep]
    diff = residuals[1][j] - residuals[0][i]
    if not len(diff):
        return {"n": 0}
    rng = np.random.default_rng(seed)
    boot = [np.median(diff[rng.integers(len(diff), size=len(diff))])
            for _ in range(1000)]
    return {"n": len(diff), "median_final_minus_cal": float(np.median(diff)),
            "bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
            "median_matching_radius": float(np.median(distance[i, j])),
            "p95_matching_radius": float(np.quantile(distance[i, j], .95))}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = np.load(BASE / "native_population.npz")
    result = {"method": "outcome-blind minimum-cost 1:1 assignment, standardized Euclidean radius <=2.0",
              "scales_and_features": {key: [list(pair) for pair in value]
                                      for key, value in FEATURE_SETS.items()},
              "matches": {key: run_match(rows, value, 2.0, 20261105 + index)
                          for index, (key, value) in enumerate(FEATURE_SETS.items())}}
    write_json(OUT, result)


if __name__ == "__main__":
    main()

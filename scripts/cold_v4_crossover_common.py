#!/usr/bin/env python3
"""Reanalyse the saved crossover draws on identical accepted sources."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from photdis.stage1 import require_compute, write_json

SOURCE = Path("data/cold_v4_crossover/rows.npz")
BASE = Path("data/cold_v4_crossover/diagnostic.json")
OUT = Path("data/cold_v4_crossover/common_acceptance.json")
N_BOOT = 2000
SEED = 20261201


def robust(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0}
    center = float(np.median(values))
    return {"n": int(len(values)), "median": center,
            "nmad": float(1.4826 * np.median(abs(values - center)))}


def paired_centers(a, b, seed, n_boot=N_BOOT):
    """Resample source identities jointly; keep both center estimands distinct."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite paired one-dimensional residuals required")
    if not len(a):
        return {"n": 0}
    center_delta = float(np.median(b) - np.median(a))
    paired_delta = float(np.median(b - a))
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(a), size=(n_boot, len(a)))
    center_draw = np.median(b[sample], axis=1) - np.median(a[sample], axis=1)
    paired_draw = np.median((b-a)[sample], axis=1)
    return {"n": int(len(a)), "environment_0": robust(a),
            "environment_1": robust(b),
            "difference_of_medians": center_delta,
            "difference_of_medians_bootstrap95": np.quantile(center_draw, [.025, .975]).tolist(),
            "median_of_paired_differences": paired_delta,
            "median_of_paired_differences_bootstrap95": np.quantile(paired_draw, [.025, .975]).tolist()}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    original = json.loads(BASE.read_text())
    report = {"source": str(SOURCE), "resample_unit": "source identity, jointly across environments",
              "bootstrap_draws": N_BOOT, "seed": SEED,
              "frozen_registration_sha256": original["registration_sha256"],
              "template_source_hashes": original["template_source_hashes"], "cells": {}}
    with np.load(SOURCE) as rows:
        for key, old in original["paired_environment"].items():
            t, d = map(int, key.split("|"))
            a, b = f"{t}|0|{d}", f"{t}|1|{d}"
            common = rows[f"{a}__paired"] & rows[f"{b}__paired"]
            if int(common.sum()) != old["common_accepted_sources"]:
                raise ValueError(f"Frozen common count differs: {key}")
            r0 = rows[f"{a}__degraded_total"][common]
            r1 = rows[f"{b}__degraded_total"][common]
            report["cells"][key] = {
                "template_field_index": t, "target_r_bin": old["target_r_bin"],
                "common": paired_centers(r0, r1, SEED + 100*t + d),
                "full_flow": {str(e): {"accepted": original["cells"][f"{t}|{e}|{d}"]["selected"],
                    "acceptance": original["cells"][f"{t}|{e}|{d}"]["acceptance"],
                    "nmad_paired_native_eligible": original["cells"][f"{t}|{e}|{d}"]["components"]["degraded_total"]["nmad"]}
                    for e in (0, 1)}}
    write_json(OUT, report)


if __name__ == "__main__":
    main()

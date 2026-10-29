#!/usr/bin/env python3
"""Check forward-fit color degeneracy, field sensitivity and negative counts."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import numpy as np

from photdis.metal_poor_forward import (design_matrix, fit,
                                        log_probability, predictive_cdf)
from photdis.stage1 import require_compute, write_json
from metal_poor_forward_experiment import prepare

SOURCE = Path("manifests/metal_poor_forward_v1.json")
OUT = Path("manifests/metal_poor_forward_diagnostics_v1.json")


def negative_count_check(data, parameters, with_feh):
    observed = int(np.sum(data["p_corr"] < 0))
    at_zero = data.copy()
    at_zero["p_corr"] = 0
    probability = predictive_cdf(parameters, at_zero, with_feh)
    expected = float(probability.sum())
    variance = float(np.sum(probability * (1 - probability)))
    return {"n": len(data), "observed": observed, "expected": expected,
            "expected_standard_deviation": float(np.sqrt(variance)),
            "standardized_difference": float((observed - expected) / np.sqrt(variance))}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = json.loads(SOURCE.read_text())
    rows, dedup = prepare()
    if dedup != prior["deduplication"]:
        raise ValueError("Forward-diagnostic sample differs from fitted sample")
    role = np.char.strip(rows["role"].astype("U"))
    field = np.char.strip(rows["field"].astype("U"))
    development = role == "development"
    reserved = role == "reserved_validation"
    x = design_matrix(rows["gr0"][development], rows["rz0"][development])
    metal = rows["feh_cal"][development] + 1
    fitted = x @ np.linalg.lstsq(x, metal, rcond=None)[0]
    r2 = 1 - np.sum((metal-fitted)**2) / np.sum((metal-metal.mean())**2)
    jackknife = {}
    for name in np.unique(field[development]):
        subset = development & (field != name)
        if subset.sum() < 100:
            continue
        params, diagnostic = fit(rows[subset], True)
        scores = log_probability(params, rows[reserved], True)
        jackknife[name] = {"fit_n": int(subset.sum()),
                           "beta": float(params[4]),
                           "intrinsic_scatter_mag": float(params[-1]),
                           "reserved_loglike_mean": float(np.mean(scores)),
                           "iterations": diagnostic["iterations"]}
    baseline = np.asarray(prior["models"]["no_feh"]["parameters"], float)
    with_metal = np.asarray(prior["models"]["one_feh"]["parameters"], float)
    result = {"source": str(SOURCE),
              "sample_counts": prior["counts"],
              "feh_explained_by_color_r2_development": float(r2),
              "negative_count": {name: {"development": negative_count_check(
                  rows[development], params, use_iron),
                  "reserved": negative_count_check(rows[reserved], params, use_iron)}
                  for name, params, use_iron in (("no_feh", baseline, False),
                                                 ("one_feh", with_metal, True))},
              "leave_one_development_field_out": jackknife,
              "compute_host": socket.gethostname()}
    write_json(OUT, result)
    print(json.dumps({"feh_color_r2": r2, "jackknife": jackknife,
                      "negative_count": result["negative_count"]}), flush=True)


if __name__ == "__main__":
    main()

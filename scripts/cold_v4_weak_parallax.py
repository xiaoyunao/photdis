#!/usr/bin/env python3
"""Forward-check frozen v4 predictions on DESI cold dwarfs with weak Gaia parallaxes."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import joblib
import numpy as np

from photdis.conditional import group_labels
from photdis.parallax_forward import parallax_forward_check
from photdis.route import FEATURES, LEVEL, apply_route_offsets, feature_matrix, route_rows
from photdis.stage1 import require_compute, write_json
from cold_v4 import CAL_CONFIG, FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED, sha
from cold_v4_type_audit import COLS, class_masks, unique_spectra

RESULT = Path("data/cold_v4_weak_parallax/forward.json")
EXTRA = ["extended_quality", "anchor", "p_corr", "gaia_parallax_error",
         "gaia_parallax", "g0"]
READ = sorted(set(COLS + EXTRA))


def summary(values):
    x = np.asarray(values, float)
    if not len(x):
        return {"n": 0}
    center = np.median(x)
    return {"n": len(x), "median": float(center),
            "nmad": float(1.4826 * np.median(np.abs(x - center))),
            "mean": float(np.mean(x))}


def metrics(observed, error, g0, predicted):
    if not len(observed):
        return {"n": 0}
    forward = parallax_forward_check(observed, error, g0, predicted)
    pit = forward["approximate_pit"]
    return {"n": len(observed),
            "negative_corrected_parallax": int(np.sum(observed < 0)),
            "observed_parallax_mas": summary(observed),
            "model_minus_observed_parallax_mas": summary(
                forward["model_parallax_mean_mas"] - observed),
            "standardized_residual": summary(forward["standardized_residual"]),
            "approximate_pit_histogram_deciles": np.histogram(
                pit, bins=np.linspace(0, 1, 11))[0].tolist(),
            "approximate_pit_below_0p05": float(np.mean(pit < .05)),
            "approximate_pit_above_0p95": float(np.mean(pit > .95))}


def field_result(config_path, route, forests, prepared):
    task = json.loads(Path(config_path).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    with fitsio.FITS(str(path)) as file:
        raw = file[1].read(columns=READ)
    table = {name: raw[name] for name in raw.dtype.names}
    unique, dedup = unique_spectra(table)
    dwarf = class_masks(table, unique & table["grz_base"])["cool_dwarf"]
    routed = route_rows(table, route)
    error = np.asarray(table["gaia_parallax_error"], float)
    parallax = np.asarray(table["p_corr"], float)
    weak = (dwarf & table["extended_quality"] & ~table["anchor"] &
            routed["selected"] & table["grz_clean"] &
            np.isfinite(parallax) & np.isfinite(error) & (error > 0) &
            (parallax / error < 10))
    row_index = np.flatnonzero(weak)
    result = {"field": task["name"], "deduplication": dedup,
              "qualified_cool_dwarf_in_grz_base": int(dwarf.sum()),
              "weak_clean_routed": len(row_index),
              "raw_negative_parallax": int(np.sum(table["gaia_parallax"][row_index] < 0)),
              "corrected_negative_parallax": int(np.sum(parallax[row_index] < 0)),
              "by_tier": {}, "by_corrected_snr": {}}
    if not len(row_index):
        return result
    pred = np.full((len(row_index), 5), np.nan)
    for tier, level in LEVEL.items():
        take = routed["tier"][row_index] == level
        if take.any():
            pred[take] = forests[tier].predict(feature_matrix(table, tier)[row_index[take]])
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite weak-parallax distance prediction")
    edges = np.asarray(prepared["route_groups"]["color_edges"], float)
    groups = group_labels(table["rz0"][row_index], table["r"][row_index],
                          edges, prepared["route_groups"]["r_edges_by_color"])
    calibrated = apply_route_offsets(pred, groups, prepared["calibration_offsets"])
    g0 = np.asarray(table["g0"][row_index], float)
    result["all"] = metrics(parallax[row_index], error[row_index], g0, calibrated)
    for tier, level in LEVEL.items():
        take = routed["tier"][row_index] == level
        result["by_tier"][tier] = metrics(parallax[row_index][take], error[row_index][take],
                                          g0[take], calibrated[take])
    snr = parallax[row_index] / error[row_index]
    for label, lo, hi in (("negative", -np.inf, 0), ("zero_to_five", 0, 5),
                          ("five_to_ten", 5, 10)):
        take = (snr >= lo) & (snr < hi)
        result["by_corrected_snr"][label] = metrics(
            parallax[row_index][take], error[row_index][take], g0[take], calibrated[take])
    print(json.dumps({"field": task["name"], "weak": len(row_index),
                      "negative": result["corrected_negative_parallax"]}), flush=True)
    return result


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Cold-v4 preparation changed")
    prepared = json.loads(PREPARED.read_text())
    route_path = OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen route changed")
    route = joblib.load(route_path)
    forests = {}
    for tier in FEATURES:
        path = OUT / f"{tier}.joblib"
        if sha(path) != prepared["tier_training"][tier]["model_sha256"]:
            raise ValueError(f"Frozen forest changed: {tier}")
        forests[tier] = joblib.load(path)
    result = {"prepared_sha256": sha(PREPARED),
              "compute_host": socket.gethostname(),
              "selection": "unique qualified DESI 3000-5000 K logg>=3.5; Gaia extended_quality; not strict anchor; corrected parallax/error<10 including negative; v4 route and grz_clean",
              "forward_model": "interpolate five calibrated Mg quantiles, map to positive parallax, convolve with Gaia Gaussian formal error; approximate PIT only",
              "limitations": "DESI targeting and Gaia quality selection may depend on parallax; Gaia zero-point errors are shared, and five quantiles do not specify a full stellar PDF",
              "fields": {}}
    for config in (CAL_CONFIG, *FINAL_CONFIGS):
        row = field_result(config, route, forests, prepared)
        result["fields"][row["field"]] = row
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, result)


if __name__ == "__main__":
    main()

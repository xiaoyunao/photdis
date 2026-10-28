#!/usr/bin/env python3
"""Selection-aware weak-Gaia-parallax audit of frozen cold-v4 forest weights."""
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
from photdis.parallax_forward import (parallax_forward_check,
                                       weighted_selected_parallax_check)
from photdis.route import FEATURES, LEVEL, apply_route_offsets, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import CAL_CONFIG, FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED, sha
from cold_v4_type_audit import COLS, class_masks, unique_spectra

LOCK = Path("configs/cold_v4_deep_policy.json")
RESULT = Path("data/cold_v4_weak_parallax/forward_v2.json")
EXTRA = ["extended_quality", "anchor", "p_corr", "gaia_parallax_error",
         "gaia_parallax", "g0", "snr_w1", "snr_w2"]
READ = sorted(set(COLS + EXTRA))


def summary(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0}
    median = float(np.median(values))
    return {"n": len(values), "median": median,
            "nmad": float(1.4826 * np.median(np.abs(values - median))),
            "mean": float(np.mean(values))}


def metrics(observed, forward, historical_pit):
    n = len(observed)
    if not n:
        return {"n": 0}
    raw = forward["unconditional_pit"]
    selected = forward["selected_pit"]
    probability = forward["selection_probability"]
    return {"n": n, "negative_corrected_parallax": int(np.sum(observed < 0)),
            "model_minus_observed_parallax_mas": summary(
                forward["model_parallax_mean_mas"] - observed),
            "standardized_residual": summary(forward["standardized_residual"]),
            "forest_unconditional_pit_below_0p05": float(np.mean(raw < .05)),
            "forest_selected_pit_below_0p05": float(np.mean(selected < .05)),
            "forest_selected_pit_above_0p95": float(np.mean(selected > .95)),
            "forest_selected_pit_deciles": np.histogram(
                selected, bins=np.linspace(0, 1, 11))[0].tolist(),
            "expected_unconditional_low_pit_given_cut": float(np.mean(
                np.minimum(.05 / probability, 1))),
            "selection_probability": summary(probability),
            "historical_cqr_five_point_approx_pit_below_0p05": float(
                np.mean(historical_pit < .05))}


def subset_forward(forward, mask):
    return {key: value[mask] for key, value in forward.items()}


def evaluate_policy(table, index, levels, forests, prepared):
    chosen = levels[index]
    keep = chosen > 0
    index = index[keep]
    chosen = chosen[keep]
    n = len(index)
    predicted = np.empty((n, 5), float)
    weights_forward = {key: np.empty(n, float) for key in (
        "model_parallax_mean_mas", "model_parallax_variance_mas2",
        "standardized_residual", "unconditional_pit", "selection_probability",
        "selected_pit")}
    observed = np.asarray(table["p_corr"][index], float)
    error = np.asarray(table["gaia_parallax_error"][index], float)
    g0 = np.asarray(table["g0"][index], float)
    for tier, number in LEVEL.items():
        take = chosen == number
        if not take.any():
            continue
        features = feature_matrix(table, tier)[index[take]]
        forest = forests[tier]
        predicted[take] = forest.predict(features)
        weights = forest.predict_weights(features)
        forward = weighted_selected_parallax_check(
            observed[take], error[take], g0[take], forest.y, weights)
        for key in weights_forward:
            weights_forward[key][take] = forward[key]
    edges = np.asarray(prepared["route_groups"]["color_edges"], float)
    groups = group_labels(table["rz0"][index], table["r"][index],
                          edges, prepared["route_groups"]["r_edges_by_color"])
    calibrated = apply_route_offsets(predicted, groups,
                                     prepared["calibration_offsets"])
    historical_pit = parallax_forward_check(observed, error, g0, calibrated)[
        "approximate_pit"]
    all_metrics = metrics(observed, weights_forward, historical_pit)
    by_tier = {tier: metrics(observed[chosen == number],
                             subset_forward(weights_forward, chosen == number),
                             historical_pit[chosen == number])
               for tier, number in LEVEL.items()}
    by_snr = {}
    snr = observed / error
    for name, lo, hi in (("negative", -np.inf, 0), ("zero_to_five", 0, 5),
                         ("five_to_ten", 5, 10)):
        take = (snr >= lo) & (snr < hi)
        by_snr[name] = metrics(observed[take], subset_forward(weights_forward, take),
                               historical_pit[take])
    return {"selected_index": index, "all": all_metrics,
            "by_tier": by_tier, "by_corrected_snr": by_snr,
            "forward_arrays": weights_forward, "historical_pit": historical_pit,
            "observed": observed}


def paired_comparison(old, new):
    old_index, new_index = old["selected_index"], new["selected_index"]
    common, i_old, i_new = np.intersect1d(old_index, new_index, return_indices=True)
    return {"n_same_objects": len(common),
            "old_route": metrics(old["observed"][i_old],
                                 subset_forward(old["forward_arrays"], i_old),
                                 old["historical_pit"][i_old]),
            "wise20_route": metrics(new["observed"][i_new],
                                    subset_forward(new["forward_arrays"], i_new),
                                    new["historical_pit"][i_new])}


def field_result(config_path, route, forests, prepared, policy):
    task = json.loads(Path(config_path).read_text())
    raw = fitsio.read(Path(task["output_dir"]) / "field_diagnostics.fits",
                      columns=READ)
    table = {name: raw[name] for name in raw.dtype.names}
    unique, dedup = unique_spectra(table)
    dwarf = class_masks(table, unique & table["grz_base"])["cool_dwarf"]
    error = np.asarray(table["gaia_parallax_error"], float)
    observed = np.asarray(table["p_corr"], float)
    base = (dwarf & table["extended_quality"] & ~table["anchor"] &
            table["grz_clean"] & np.isfinite(observed) & np.isfinite(error) &
            (error > 0) & (observed / error < 10))
    index = np.flatnonzero(base)
    routed = route_rows(table, route)
    old_level = np.where(table["grz_clean"], routed["tier"], 0)
    new_level = choose_policy(table, routed, policy)
    result = {"field": task["name"], "deduplication": dedup,
              "qualified_cool_dwarf_in_grz_base": int(dwarf.sum()),
              "weak_dwarf_before_route": len(index),
              "raw_negative_before_route": int(np.sum(table["gaia_parallax"][index] < 0)),
              "corrected_negative_before_route": int(np.sum(observed[index] < 0))}
    if not len(index):
        result.update(old_route={"n": 0}, wise20_route={"n": 0},
                      same_object_comparison={"n_same_objects": 0})
        return result
    old = evaluate_policy(table, index, old_level, forests, prepared)
    new = evaluate_policy(table, index, new_level, forests, prepared)
    result["old_route"] = {key: old[key] for key in ("all", "by_tier", "by_corrected_snr")}
    result["wise20_route"] = {key: new[key] for key in ("all", "by_tier", "by_corrected_snr")}
    result["same_object_comparison"] = paired_comparison(old, new)
    print(json.dumps({"field": task["name"], "before_route": len(index),
                      "old": old["all"]["n"], "wise20": new["all"]["n"]}), flush=True)
    return result


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Cold-v4 preparation changed")
    prepared = json.loads(PREPARED.read_text())
    lock = json.loads(LOCK.read_text())
    if lock["prepared_sha256"] != sha(PREPARED):
        raise ValueError("WISE20 lock differs from frozen v4")
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
              "policy_lock_sha256": sha(LOCK), "compute_host": socket.gethostname(),
              "distribution": "forest leaf-weighted discrete empirical Mg labels, no CQR; Gaussian Gaia formal parallax error convolution",
              "selection_condition": "observed zero-point-corrected parallax / formal error < 10; PIT is F(y)/F(10sigma)",
              "selection_scope": "explicit parallax threshold only; DESI/Gaia targeting and quality cuts are additional conditioning",
              "historical_comparator": "CQR-expanded five-quantile interpolation, old approximate unconditional PIT, scored on same objects only",
              "fields": {}}
    for config in (CAL_CONFIG, *FINAL_CONFIGS):
        row = field_result(config, route, forests, prepared, lock["chosen_policy"])
        result["fields"][row.pop("field")] = row
    write_json(RESULT, result)


if __name__ == "__main__":
    main()

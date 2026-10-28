#!/usr/bin/env python3
"""Calibrate final v4 distance-modulus intervals on disjoint simulated sky fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import numpy as np

from photdis.followup import apply_cqr, cqr_offset, identity_hash
from photdis.noise import interval_diagnostics, simulate_shared_flux, transformed_distance_quantiles
from photdis.photometry import derive_photometry
from photdis.route import FEATURES, LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import PREPARED, sha
from cold_v4_policy_noise import (REGISTRATION, frozen_models, load_environment,
                                  raw_for_ids, source_templates)
from noise_injection_v2 import source_cluster_ci

LOCK = Path("configs/cold_v4_deep_policy.json")
CAL_RESULT = Path("data/cold_v4_deep/calibration_v2.json")
CAL_TRACKED = Path("manifests/cold_v4_deep_calibration_v2.json")
FINAL_RESULT = Path("data/cold_v4_deep/final_v2.json")
MIN_GROUP = 80


def used_sigma_and_support(phot, route, tier, use):
    bands = ("g", "r", "z") + (("w1",) if LEVEL[tier] >= 2 else ()) + (
        ("w2",) if LEVEL[tier] == 3 else ())
    sigma = np.maximum.reduce([np.asarray(phot[f"sigma_{band}"][use], float)
                               for band in bands])
    model = route["tiers"][tier]
    distance = model["tree"].query(feature_matrix(phot, tier)[use] / model["scale"],
                                   k=route["neighbors"])[0][:, -1]
    return sigma, distance / model["radius"]


def draw_field(field, config, policy, repeats, rng, route, forests):
    template, truth_mg, truth_mu, source_id, metadata = source_templates(field, rng)
    mother = load_environment(field)
    stage = json.loads(Path("configs/stage1.json").read_text())
    prepared = json.loads(PREPARED.read_text())
    edges = np.asarray(prepared["route_groups"]["color_edges"], float)
    records = []
    for depth, (lo, hi) in enumerate(config["depth_bins_r"]):
        pool = mother[mother["grz_base"] & (mother["r"] >= lo) & (mother["r"] < hi)]
        if len(pool) < 100:
            raise ValueError(f"Only {len(pool)} grz target environments at r={lo}-{hi}")
        source_index = np.repeat(np.arange(len(template)), repeats)
        target_ids = np.asarray(pool["ls_id"][rng.integers(len(pool), size=len(source_index))], np.int64)
        unique, inverse = np.unique(target_ids, return_inverse=True)
        target = raw_for_ids(unique, field)[inverse]
        raw, shift, noise_ratio = simulate_shared_flux(template[source_index], target, rng)
        phot = derive_photometry(raw, **stage["photometry"])
        routed = route_rows(phot, route)
        level = choose_policy(phot, routed, policy)
        observed_r = np.asarray(phot["r"], float)
        application_depth = np.isfinite(observed_r) & (observed_r >= 18) & (observed_r < 22)
        level[~application_depth] = 0
        selected = level > 0
        observed_depth_index = np.searchsorted([19, 20, 21], observed_r, side="right")
        predicted_mg = np.full((len(level), 5), np.nan)
        sigma = np.full(len(level), np.nan)
        support = np.full(len(level), np.nan)
        for tier, number in LEVEL.items():
            use = level == number
            if use.any():
                predicted_mg[use] = forests[tier].predict(feature_matrix(phot, tier)[use])
                sigma[use], support[use] = used_sigma_and_support(phot, route, tier, use)
        if not np.isfinite(predicted_mg[selected]).all():
            raise ValueError("Selected prediction has nonfinite quantiles")
        predicted_mu = np.full_like(predicted_mg, np.nan)
        predicted_mu[selected] = transformed_distance_quantiles(
            np.asarray(phot["g0"][selected], float), predicted_mg[selected])[0]
        records.append({
            "depth_index": depth, "r_bin": [lo, hi], "target_mother_rows": len(pool),
            "template_sources": len(template), "draws": len(level),
            "grz_base": int(phot["grz_base"].sum()),
            "grz_clean": int(phot["grz_clean"].sum()),
            "observed_r_18_to_22": int(application_depth.sum()),
            "selected": selected, "level": level,
            "observed_depth_index": observed_depth_index,
            "source_index": source_index,
            "truth_mg": truth_mg[source_index],
            "truth_mu": truth_mu[source_index] + shift,
            "g0": np.asarray(phot["g0"], float),
            "predicted_mg": predicted_mg,
            "predicted_mu": predicted_mu,
            "color_group": np.searchsorted(edges[1:-1],
                                           np.asarray(phot["rz0"], float), side="right"),
            "error_group": (sigma > .10).astype(np.uint8),
            "support_group": (support > .8).astype(np.uint8),
            "template_noise_gt_target": {
                band: float(np.nanmean(noise_ratio[:, index] > 1))
                for index, band in enumerate(("g", "r", "z", "w1", "w2"))},
        })
        print(f"{metadata['name']} r={lo}-{hi}: {int(selected.sum())}/{len(selected)} selected", flush=True)
    return records, source_id, metadata


def group_key(depth, tier=None, color=None, error=None, support=None):
    values = (depth, tier, color, error, support)
    return "|".join("*" if value is None else str(value) for value in values)


def offset_candidates(row, index):
    depth = int(row["observed_depth_index"][index])
    tier = int(row["level"][index])
    color = int(row["color_group"][index])
    error = int(row["error_group"][index])
    support = int(row["support_group"][index])
    return (group_key(depth, tier, color, error, support),
            group_key(depth, tier, color),
            group_key(depth, tier),
            group_key(depth))


def fit_offsets(records):
    """Use one independent source per observed-r bin; pool sparse conditions."""
    members = {}
    seen_source_bin = set()
    for row in records:
        if row["draws"] != row["template_sources"]:
            raise ValueError("Calibration requires one draw per source and depth")
        for index in np.flatnonzero(row["selected"]):
            unique_key = (int(row["observed_depth_index"][index]),
                          int(row["source_index"][index]))
            if unique_key in seen_source_bin:
                continue
            seen_source_bin.add(unique_key)
            for key in offset_candidates(row, index):
                members.setdefault(key, []).append((row, index))
    offsets = {}
    for key, rows in members.items():
        n = len(rows)
        if n < 19 or (n < MIN_GROUP and key.count("*") < 4):
            continue
        truth = np.array([row["truth_mu"][i] for row, i in rows])
        predicted = np.array([row["predicted_mu"][i] for row, i in rows])
        q68, _ = cqr_offset(truth, predicted, 1, 3, .32)
        q95, _ = cqr_offset(truth, predicted, 0, 4, .05)
        if not np.isfinite(q68) or not np.isfinite(q95):
            raise ValueError("Nonfinite split-CQR offset")
        offsets[key] = {"n_independent_sources": n,
                        "q68": q68, "q95": q95}
    return offsets


def calibrated_mu(row, offsets):
    adjusted = row["predicted_mu"].copy()
    source_count = {}
    for index in np.flatnonzero(row["selected"]):
        key = next((key for key in offset_candidates(row, index) if key in offsets), None)
        if key is None:
            raise ValueError("No depth-global CQR offset for selected row")
        source_count[key] = source_count.get(key, 0) + 1
        adjusted[index:index + 1] = apply_cqr(
            adjusted[index:index + 1], offsets[key]["q68"], offsets[key]["q95"])
    return adjusted, source_count


def summarize(row, adjusted, seed):
    selected = row["selected"]
    report = {"r_bin": row["r_bin"],
              "target_mother_rows": row["target_mother_rows"],
              "independent_template_sources": row["template_sources"],
              "draws": row["draws"],
              "grz_base": row["grz_base"], "grz_clean": row["grz_clean"],
              "observed_r_18_to_22": row["observed_r_18_to_22"],
              "selected": int(selected.sum()),
              "acceptance": float(selected.mean()),
              "selected_by_tier": {tier: int(np.sum(row["level"] == number))
                                   for tier, number in LEVEL.items()},
              "template_noise_gt_target_fraction": row["template_noise_gt_target"],
              "tiers": {}, "strata": {}}
    report["selected_by_observed_r_bin"] = {
        str(index): int(np.sum(selected & (row["observed_depth_index"] == index)))
        for index in range(4)}
    for name, use in [("all", selected), *[(tier, row["level"] == number)
                                          for tier, number in LEVEL.items()]]:
        if not use.any():
            continue
        truth = row["truth_mu"][use]
        direct = row["predicted_mu"][use]
        cal = adjusted[use]
        index = row["source_index"][use]
        # Apply interval diagnostics directly to final distance modulus.
        report["tiers"][name] = {
            "n_draws": int(use.sum()),
            "n_sources": int(len(np.unique(index))),
            "direct_mu": interval_diagnostics(truth, direct),
            "calibrated_mu": interval_diagnostics(truth, cal),
            "calibrated_source_cluster_ci": source_cluster_ci(
                truth, cal, index, seed=seed + len(report["tiers"]), n_boot=200),
        }
    conditions = {"color_tertile": range(3), "used_error_gt_0p10": range(2),
                  "support_ratio_gt_0p8": range(2)}
    arrays = {"color_tertile": row["color_group"],
              "used_error_gt_0p10": row["error_group"],
              "support_ratio_gt_0p8": row["support_group"]}
    for name, labels in conditions.items():
        report["strata"][name] = {}
        for value in labels:
            use = selected & (arrays[name] == value)
            if use.sum() >= 20:
                report["strata"][name][str(value)] = interval_diagnostics(
                    row["truth_mu"][use], adjusted[use])
            else:
                report["strata"][name][str(value)] = {"n": int(use.sum())}
    report["strata"]["tier_x_condition"] = {}
    for tier, level in LEVEL.items():
        report["strata"]["tier_x_condition"][tier] = {}
        for name, labels in conditions.items():
            report["strata"]["tier_x_condition"][tier][name] = {}
            for value in labels:
                use = selected & (row["level"] == level) & (arrays[name] == value)
                report["strata"]["tier_x_condition"][tier][name][str(value)] = (
                    interval_diagnostics(row["truth_mu"][use], adjusted[use])
                    if use.sum() >= 20 else {"n": int(use.sum())})
    return report


def summarize_observed_bins(rows, adjusted_rows, seed):
    """Pool experimental depth strata by the r bin seen by the application."""
    result = {}
    for depth in range(4):
        pooled = {}
        for name in ("truth_mu", "predicted_mu", "calibrated_mu", "source_index", "level"):
            pieces = []
            for row, adjusted in zip(rows, adjusted_rows):
                use = row["selected"] & (row["observed_depth_index"] == depth)
                array = adjusted if name == "calibrated_mu" else row[name]
                pieces.append(array[use])
            pooled[name] = np.concatenate(pieces)
        entry = {"observed_r_bin": [18 + depth, 19 + depth],
                 "n_draws": len(pooled["truth_mu"]),
                 "n_sources": int(len(np.unique(pooled["source_index"]))),
                 "tiers": {}}
        for name, use in [("all", np.ones(entry["n_draws"], bool)),
                          *[(tier, pooled["level"] == number)
                            for tier, number in LEVEL.items()]]:
            if not use.any():
                continue
            truth = pooled["truth_mu"][use]
            direct = pooled["predicted_mu"][use]
            cal = pooled["calibrated_mu"][use]
            sources = pooled["source_index"][use]
            entry["tiers"][name] = {
                "n_draws": len(truth),
                "n_sources": int(len(np.unique(sources))),
                "direct_mu": interval_diagnostics(truth, direct),
                "calibrated_mu": interval_diagnostics(truth, cal),
                "calibrated_source_cluster_ci": source_cluster_ci(
                    truth, cal, sources, seed=seed + depth + len(entry["tiers"]), n_boot=200),
            }
        result[str(depth)] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "final"))
    args = parser.parse_args()
    require_compute()
    output = CAL_RESULT if args.mode == "calibrate" else FINAL_RESULT
    if output.exists():
        raise FileExistsError(output)
    registration = json.loads(REGISTRATION.read_text())
    policy_lock = json.loads(LOCK.read_text())
    if policy_lock["registration_sha256"] != sha(REGISTRATION):
        raise ValueError("Deep field registration changed")
    if policy_lock["prepared_sha256"] != sha(PREPARED):
        raise ValueError("Frozen v4 model changed")
    if policy_lock["policy_development_sha256"] != sha(
            "manifests/cold_v4_policy_development.json"):
        raise ValueError("Locked policy development result changed")
    policy = policy_lock["chosen_policy"]
    route, forests = frozen_models()
    role = "interval_calibration" if args.mode == "calibrate" else "independent_final"
    field = registration[role]["field_config"]
    rng = np.random.default_rng(registration["seed"] + (101 if args.mode == "calibrate" else 202))
    repeats = 1 if args.mode == "calibrate" else 4
    rows, source_id, metadata = draw_field(field, registration, policy, repeats, rng,
                                           route, forests)
    if args.mode == "calibrate":
        offsets = fit_offsets(rows)
    else:
        if sha(CAL_RESULT) != sha(CAL_TRACKED):
            raise ValueError("Published deep calibration changed")
        calibration = json.loads(CAL_RESULT.read_text())
        if calibration["policy_lock_sha256"] != sha(LOCK):
            raise ValueError("Deep policy lock changed after calibration")
        if calibration["registration_sha256"] != sha(REGISTRATION):
            raise ValueError("Deep registration changed after calibration")
        offsets = calibration["offsets"]
    result = {"role": role, "field": metadata,
              "compute_host": socket.gethostname(),
              "registration_sha256": sha(REGISTRATION),
              "policy_lock_sha256": sha(LOCK),
              "prepared_sha256": sha(PREPARED),
              "chosen_policy": policy,
              "independent_template_sources": len(source_id),
              "template_source_id_sha256": identity_hash(source_id),
              "repeats_per_source_and_depth": repeats,
              "calibration_unit": "one independent source per measured-r bin; repeated source crossings are deduplicated within each measured-r bin; repeats only in final",
              "condition_hierarchy": "measured-r-bin+tier+color_tertile+used_error_0p10+support_0p8; sparse groups pool to measured-r-bin+tier+color, measured-r-bin+tier, measured-r-bin",
              "min_group_independent_sources": MIN_GROUP,
              "offsets": offsets if args.mode == "calibrate" else None,
              "depths": []}
    adjusted_rows = []
    for row in rows:
        adjusted, applied = calibrated_mu(row, offsets)
        adjusted_rows.append(adjusted)
        summary = summarize(row, adjusted, registration["seed"] + row["depth_index"])
        summary["applied_offset_group_counts"] = applied
        result["depths"].append(summary)
    result["observed_r_bins"] = summarize_observed_bins(
        rows, adjusted_rows, registration["seed"] + 700)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose native/degraded residuals with a 2x2 source-field/noise-field swap."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.followup import identity_hash
from photdis.noise import simulate_shared_flux
from photdis.photometry import derive_photometry
from photdis.route import FEATURES, LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import PHOT_COLUMNS, require_compute, write_json
from cold_v4 import PREPARED, sha
from cold_v4_deep_calibration import LOCK, used_sigma_and_support
from cold_v4_policy_noise import REGISTRATION, frozen_models, raw_for_ids, source_templates

OUT = Path("data/cold_v4_crossover")
MANIFEST = OUT / "diagnostic.json"
ROWS = OUT / "rows.npz"
SOURCE_COUNT = 500
SEED = 20261130
N_BOOT = 400


def robust(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0}
    center = float(np.median(values))
    return {"n": len(values), "median": center,
            "nmad": float(1.4826 * np.median(np.abs(values - center))),
            "p05": float(np.quantile(values, .05)),
            "p95": float(np.quantile(values, .95))}


def median_ci(values, seed, n_boot=N_BOOT):
    values = np.asarray(values, float)
    if len(values) < 20:
        return None
    rng = np.random.default_rng(seed)
    draws = [np.median(values[rng.integers(len(values), size=len(values))])
             for _ in range(n_boot)]
    return np.quantile(draws, [.025, .975]).tolist()


def decompose(native_mu, native_same_tier_mu, degraded_mu, truth_mu, shift):
    """Pair each star before reporting any population-level summary."""
    native = native_mu - truth_mu
    total = degraded_mu - truth_mu - shift
    increment = degraded_mu - native_mu - shift
    route_change = native_same_tier_mu - native_mu
    fixed_tier_noise = degraded_mu - native_same_tier_mu - shift
    if not np.allclose(total, native + increment, atol=1e-10):
        raise ValueError("Native/degraded decomposition failed")
    if not np.allclose(increment, route_change + fixed_tier_noise,
                       atol=1e-10, equal_nan=True):
        raise ValueError("Tier-change decomposition failed")
    return {"native_residual": native, "degraded_total": total,
            "degradation_increment": increment, "route_change": route_change,
            "fixed_tier_noise": fixed_tier_noise}


def upper_raw(table):
    columns = ["MASKBITS", *PHOT_COLUMNS]
    raw = np.empty(len(table), dtype=[(name, table[name.lower()].dtype)
                                      for name in columns])
    for name in columns:
        raw[name] = table[name.lower()]
    return raw


def source_set(field, seed, route, forests, phot_config, policy):
    template, truth_mg, truth_mu, source_id, metadata = source_templates(
        field, np.random.default_rng(seed))
    if len(template) != SOURCE_COUNT:
        raise ValueError(f"Expected {SOURCE_COUNT} source templates")
    phot = derive_photometry(upper_raw(template), **phot_config)
    routed = route_rows(phot, route)
    level = choose_policy(phot, routed, policy)
    predictions = {}
    for tier, number in LEVEL.items():
        use = routed["support_by_tier"][tier]
        values = np.full(len(template), np.nan)
        if use.any():
            predicted = forests[tier].predict(feature_matrix(phot, tier)[use])
            values[use] = phot["g0"][use] - predicted[:, 2]
        predictions[tier] = values
    chosen_mu = np.full(len(template), np.nan)
    for tier, number in LEVEL.items():
        use = level == number
        chosen_mu[use] = predictions[tier][use]
    if not np.isfinite(chosen_mu[level > 0]).all():
        raise ValueError("Native selected prediction is nonfinite")
    return {"template": template, "truth_mu": truth_mu,
            "truth_mg": truth_mg, "source_id": source_id,
            "metadata": metadata, "phot": phot, "level": level,
            "predictions": predictions, "chosen_mu": chosen_mu,
            "source_identity_sha256": identity_hash(source_id)}


def environment_mother(field):
    config = json.loads(Path(field).read_text())
    path = Path(config["output_dir"]) / "field_diagnostics.fits"
    return fitsio.read(path, columns=["ls_id", "r", "r0", "grz_base"])


def nearest_ids(pool, requested_r0):
    order = np.argsort(pool["r0"])
    values = np.asarray(pool["r0"][order], float)
    pos = np.searchsorted(values, requested_r0)
    lo = np.clip(pos - 1, 0, len(values) - 1)
    hi = np.clip(pos, 0, len(values) - 1)
    chosen = np.where(abs(values[lo] - requested_r0) <=
                      abs(values[hi] - requested_r0), lo, hi)
    return np.asarray(pool["ls_id"][order[chosen]], np.int64), values[chosen]


def target_sets(fields, depth_bins, source_count, seed):
    mothers = [environment_mother(field) for field in fields]
    rng = np.random.default_rng(seed)
    ids = {0: [], 1: []}
    matching = []
    for lo, hi in depth_bins:
        pools = [mother[mother["grz_base"] & np.isfinite(mother["r0"]) &
                         (mother["r"] >= lo) & (mother["r"] < hi)]
                 for mother in mothers]
        if min(map(len, pools)) < 1000:
            raise ValueError(f"Insufficient grz noise rows at r={lo}-{hi}")
        low = max(float(np.quantile(pool["r0"], .01)) for pool in pools)
        high = min(float(np.quantile(pool["r0"], .99)) for pool in pools)
        if high - low < .5:
            raise ValueError("Target r0 distributions do not overlap adequately")
        requested = rng.uniform(low, high, source_count)
        chosen = [nearest_ids(pool, requested) for pool in pools]
        for env in (0, 1):
            ids[env].append(chosen[env][0])
        matching.append({"target_r_bin": [lo, hi], "requested_r0": [low, high],
                         "environment_pool_rows": [len(pool) for pool in pools],
                         "median_abs_target_r0_difference": float(np.median(
                             np.abs(chosen[0][1] - chosen[1][1]))),
                         "p95_abs_target_r0_difference": float(np.quantile(
                             np.abs(chosen[0][1] - chosen[1][1]), .95)),
                         "unique_target_rows": [len(np.unique(item[0])) for item in chosen]})
    raw = {}
    for env, field in enumerate(fields):
        all_ids = np.concatenate(ids[env])
        unique, inverse = np.unique(all_ids, return_inverse=True)
        found = raw_for_ids(unique, field)
        ordered = found[inverse]
        raw[env] = [ordered[i*source_count:(i+1)*source_count]
                    for i in range(len(depth_bins))]
    return raw, matching


def score_cell(source, target, route, forests, phot_config, policy, rng):
    raw, shift, noise_ratio = simulate_shared_flux(source["template"], target, rng)
    phot = derive_photometry(raw, **phot_config)
    routed = route_rows(phot, route)
    level = choose_policy(phot, routed, policy)
    measured_r = np.asarray(phot["r"], float)
    level[~(np.isfinite(measured_r) & (measured_r >= 18) & (measured_r < 22))] = 0
    native_level = source["level"]
    selected = level > 0
    mu = np.full(len(level), np.nan)
    same_native = np.full(len(level), np.nan)
    sigma = np.full(len(level), np.nan)
    support = np.full(len(level), np.nan)
    for tier, number in LEVEL.items():
        use = selected & (level == number)
        if use.any():
            predicted = forests[tier].predict(feature_matrix(phot, tier)[use])
            mu[use] = np.asarray(phot["g0"][use], float) - predicted[:, 2]
            same_native[use] = source["predictions"][tier][use]
            sigma[use], support[use] = used_sigma_and_support(phot, route, tier, use)
    paired = selected & (native_level > 0) & np.isfinite(same_native)
    vectors = {name: np.full(len(level), np.nan) for name in (
        "native_residual", "degraded_total", "degradation_increment",
        "route_change", "fixed_tier_noise")}
    if paired.any():
        pieces = decompose(source["chosen_mu"][paired], same_native[paired],
                           mu[paired], source["truth_mu"][paired], shift[paired])
        for name, value in pieces.items():
            vectors[name][paired] = value
    vectors.update({"selected": selected, "paired": paired, "level": level,
                    "native_level": native_level, "measured_r": measured_r,
                    "sigma_used": sigma, "support_ratio": support,
                    "target_ext_g": -2.5 * np.log10(np.asarray(
                        target["mw_transmission_g"], float)),
                    "source_ext_g": -2.5 * np.log10(np.asarray(
                        source["template"]["mw_transmission_g"], float)),
                    "native_rz": np.asarray(source["phot"]["rz0"], float),
                    "native_gr": np.asarray(source["phot"]["gr0"], float),
                    "shift": shift,
                    "noise_ratio_g": noise_ratio[:, 0]})
    return vectors


def cell_summary(row, source_count, seed):
    selected = row["selected"]
    paired = row["paired"]
    result = {"n_templates": source_count, "selected": int(selected.sum()),
              "selected_with_native_prediction": int(paired.sum()),
              "acceptance": float(selected.mean()),
              "tier_counts": {tier: int(np.sum(row["level"] == number))
                              for tier, number in LEVEL.items()},
              "tier_switch_count": int(np.sum(paired &
                  (row["level"] != row["native_level"]))),
              "same_tier_native_supported": int(np.sum(paired)),
              "sigma_used": robust(row["sigma_used"][paired]),
              "sigma_gt_0p10": int(np.sum(row["sigma_used"][paired] > .10)),
              "support_ratio": robust(row["support_ratio"][paired]),
              "native_rz": robust(row["native_rz"][paired]),
              "source_ext_g": robust(row["source_ext_g"][paired]),
              "target_ext_g": robust(row["target_ext_g"][paired]),
              "components": {}, "by_tier": {}, "by_switch": {}}
    for name in ("native_residual", "degraded_total", "degradation_increment",
                 "route_change", "fixed_tier_noise"):
        values = row[name][paired]
        result["components"][name] = robust(values)
        result["components"][name]["median_source_bootstrap95"] = median_ci(
            values, seed + len(result["components"]))
    for tier, number in LEVEL.items():
        use = paired & (row["level"] == number)
        result["by_tier"][tier] = {"n": int(use.sum()),
                                    "total": robust(row["degraded_total"][use]),
                                    "increment": robust(row["degradation_increment"][use])}
    for name, use in (("unchanged", paired & (row["level"] == row["native_level"])),
                      ("fallback", paired & (row["level"] < row["native_level"]))):
        result["by_switch"][name] = {
            "n": int(use.sum()),
            "route_change": robust(row["route_change"][use]),
            "fixed_tier_noise": robust(row["fixed_tier_noise"][use])}
    return result


def main():
    require_compute()
    if MANIFEST.exists() or ROWS.exists():
        raise FileExistsError("Cross experiment already exists")
    registration = json.loads(REGISTRATION.read_text())
    lock = json.loads(LOCK.read_text())
    if lock["registration_sha256"] != sha(REGISTRATION) or (
            lock["prepared_sha256"] != sha(PREPARED)):
        raise ValueError("Frozen field/model/policy registration changed")
    fields = (registration["interval_calibration"]["field_config"],
              registration["independent_final"]["field_config"])
    route, forests = frozen_models()
    stage = json.loads(Path("configs/stage1.json").read_text())
    sources = [source_set(field, registration["seed"] + seed, route, forests,
                          stage["photometry"], lock["chosen_policy"])
               for field, seed in zip(fields, (101, 202))]
    original = [json.loads(Path(path).read_text()) for path in (
        "manifests/cold_v4_deep_calibration_v2.json",
        "manifests/cold_v4_deep_final_v2.json")]
    for source, old in zip(sources, original):
        if source["source_identity_sha256"] != old["template_source_id_sha256"]:
            raise ValueError("Diagnostic template identity differs from frozen experiment")
    targets, matching = target_sets(fields, registration["depth_bins_r"],
                                     SOURCE_COUNT, SEED + 400)
    rows = {}
    summary = {"method": "2x2 previously inspected fields; same template sources as frozen v2, common requested r0 across noise environments; development diagnostic only",
               "prepared_sha256": sha(PREPARED), "policy_lock_sha256": sha(LOCK),
               "registration_sha256": sha(REGISTRATION),
               "compute_host": socket.gethostname(),
               "template_fields": [source["metadata"]["name"] for source in sources],
               "template_source_hashes": [source["source_identity_sha256"] for source in sources],
               "noise_fields": [json.loads(Path(path).read_text())["name"] for path in fields],
               "matching": matching, "native": {}, "cells": {}, "paired_environment": {}}
    for t, source in enumerate(sources):
        use = source["level"] > 0
        residual = source["chosen_mu"][use] - source["truth_mu"][use]
        summary["native"][str(t)] = {
            "selected": int(use.sum()), "tier_counts": {
                tier: int(np.sum(source["level"] == number))
                for tier, number in LEVEL.items()},
            "residual": robust(residual),
            "median_source_bootstrap95": median_ci(residual, SEED + t),
            "native_rz": robust(source["phot"]["rz0"][use]),
            "source_ext_g": robust(-2.5 * np.log10(source["template"][
                "mw_transmission_g"][use]))}
        for d, depth in enumerate(registration["depth_bins_r"]):
            for e in (0, 1):
                rng = np.random.default_rng(SEED + t*10000 + d*100 + e)
                row = score_cell(source, targets[e][d], route, forests,
                                 stage["photometry"], lock["chosen_policy"], rng)
                key = f"{t}|{e}|{d}"
                rows[key] = row
                summary["cells"][key] = {"template_field_index": t,
                                          "noise_field_index": e,
                                          "target_r_bin": depth,
                                          **cell_summary(row, SOURCE_COUNT,
                                                         SEED + t*10000 + d*100 + e)}
                print(f"template {t} noise {e} r={depth}: "
                      f"{int(row['selected'].sum())}/{SOURCE_COUNT}", flush=True)
            a, b = rows[f"{t}|0|{d}"], rows[f"{t}|1|{d}"]
            common = a["paired"] & b["paired"]
            difference = b["degraded_total"][common] - a["degraded_total"][common]
            summary["paired_environment"][f"{t}|{d}"] = {
                "template_field_index": t, "target_r_bin": depth,
                "common_accepted_sources": int(common.sum()),
                "total_residual_noise_field_1_minus_0": robust(difference),
                "median_source_bootstrap95": median_ci(
                    difference, SEED + t*100 + d)}
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {f"{key}__{name}": value for key, row in rows.items()
              for name, value in row.items()}
    np.savez_compressed(ROWS, **arrays)
    write_json(MANIFEST, summary)


if __name__ == "__main__":
    main()

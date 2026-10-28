#!/usr/bin/env python3
"""Fit expanded cold-dwarf forests, calibrate their route, score new fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import healpy as hp
import joblib
import numpy as np

from photdis.audit import checked_gaia_color
from photdis.conditional import group_labels
from photdis.followup import common_nonoptical, identity_hash
from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.noise import interval_diagnostics, transformed_distance_quantiles
from photdis.quantile import QuantileForest
from photdis.route import (FEATURES, LAYER, LEVEL, apply_route_offsets,
                           feature_matrix, fit_route, fit_route_offsets, route_rows)
from photdis.stage1 import require_compute, write_json
from cold_baseline import base_sample
from conditional_calibration import GAIA, READ_COLUMNS, sky_block_ci

REGISTRATION = Path("configs/cold_v4_fields.json")
FROZEN = Path("configs/followup_fields.json")
OUT = Path("data/cold_v4")
PREPARED = OUT / "prepared.json"
PREPARED_TRACKED = Path("manifests/cold_v4_prepared.json")
FINAL = OUT / "final.json"
PRIOR_CONFIGS = ("configs/field_cal_ra140_150.json",
                 "configs/field_test_ra190_200.json",
                 "configs/field_test_ra320_330.json")
CAL_CONFIG = "configs/field_dev_cal_ra205_215.json"
FINAL_CONFIGS = ("configs/field_cold_v4_ra000_010.json",
                 "configs/field_cold_v4_ra150_160.json",
                 "configs/field_cold_v4_ra220_230.json")
COLOR_BINS = ((1.3, 1.8), (1.8, 2.3), (2.3, 2.8), (2.8, 3.5))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registration():
    config = json.loads(REGISTRATION.read_text())
    frozen = json.loads(FROZEN.read_text())
    if config["parent_frozen_baseline_commit"] != frozen["baseline_commit"]:
        raise ValueError("New registration does not match frozen baseline")
    if sha("manifests/cold_baseline.json") != frozen["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline manifest changed")
    for tier, digest in frozen["baseline_model_sha256"].items():
        if sha(Path("data/cold_baseline_v3") / f"{tier}.joblib") != digest:
            raise ValueError(f"Frozen baseline model changed: {tier}")
    return config, frozen


def configured_field(config_path, references):
    task = json.loads(Path(config_path).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    with fitsio.FITS(str(path)) as file:
        rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        raw = file[1].read(columns=READ_COLUMNS, rows=rows)
    table = {name: raw[name] for name in raw.dtype.names}
    gaia = {name[5:]: table[name] for name in GAIA}
    table.update(checked_gaia_color(gaia))
    external = annotate_holdouts(references, table["source_id"],
                                 table["ra"], table["dec"])["external_any"]
    cold, _, mg = common_nonoptical(table, external)
    cold &= table["anchor"]
    return table, cold, mg, {"name": task["name"],
                             "bounds_deg": task["bounds_deg"],
                             "path": str(path), "strict_anchor_rows": len(rows),
                             "cold_strict_anchor_rows": int(cold.sum()),
                             "cold_identity_sha256": identity_hash(table["source_id"][cold])}


def original_pilot(references):
    raw = fitsio.read("data/pilot_audit_v1/reference.fits", ext=1)
    external = annotate_holdouts(references, raw["source_id"],
                                 raw["gaia_ra"], raw["gaia_dec"])["external_any"]
    selected, mg = base_sample(raw, external)
    baseline = json.loads(Path("manifests/cold_baseline.json").read_text())
    cells = hp.ang2pix(64, raw["gaia_ra"], raw["gaia_dec"], nest=True, lonlat=True)
    mapping = baseline["split"]["cell_map"]
    partition = np.array([mapping.get(str(int(cell)), 3) for cell in cells], np.int8)
    if np.any(selected & (partition == 3)):
        raise ValueError("Frozen original split omits a cold anchor")
    table = {name: raw[name] for name in raw.dtype.names}
    fit = selected & (partition != 2)
    test = selected & (partition == 2)
    if int(test.sum()) < 200:
        raise ValueError("Original reserved cold test unexpectedly small")
    return (table, fit, mg, {"name": "original_pilot_non_test",
                             "cold_strict_anchor_rows": int(fit.sum()),
                             "cold_identity_sha256": identity_hash(table["source_id"][fit])}), \
           (table, test, mg, {"name": "original_pilot_frozen_test",
                              "cold_strict_anchor_rows": int(test.sum()),
                              "cold_identity_sha256": identity_hash(table["source_id"][test])})


def per_tier_training(sources, tier):
    arrays, labels, ids = [], [], []
    by_field = {}
    for table, selected, mg, source in sources:
        matrix = feature_matrix(table, tier)
        take = selected & np.asarray(table[LAYER[tier]], bool) & np.isfinite(matrix).all(axis=1)
        arrays.append(matrix[take])
        labels.append(np.asarray(mg[take], float))
        ids.append(np.asarray(table["source_id"][take], np.int64))
        by_field[source["name"]] = int(take.sum())
    x = np.concatenate(arrays)
    y = np.concatenate(labels)
    identity = np.concatenate(ids)
    if len(x) < 1000 or not np.isfinite(y).all():
        raise ValueError(f"Too few finite {tier} training anchors")
    return x, y, identity, by_field


def cold_color_counts(sources):
    result = {}
    for table, selected, _, source in sources:
        color = np.asarray(table["bp_rp_checked"], float)
        result[source["name"]] = {f"{lo:g}_{hi:g}": int((selected &
                                   (color >= lo) & (color < hi)).sum())
                                   for lo, hi in COLOR_BINS}
    return result


def routed_prediction(table, cold, route, forests):
    routed = route_rows(table, route)
    take = np.asarray(cold, bool) & routed["selected"]
    prediction = np.full((len(table["source_id"]), 5), np.nan)
    tier_counts = {}
    for tier, level in LEVEL.items():
        use = take & (routed["tier"] == level)
        tier_counts[tier] = int(use.sum())
        if use.any():
            prediction[use] = forests[tier].predict(feature_matrix(table, tier)[use])
    if not np.isfinite(prediction[take]).all():
        raise ValueError("Nonfinite routed forest prediction")
    return take, prediction, routed, tier_counts


def score(table, mg, take, direct, calibrated):
    truth = np.asarray(mg[take], float)
    g0 = np.asarray(table["g0"][take], float)
    parallax = np.asarray(table["p_corr"][take], float)
    mu_truth = 10 - 5 * np.log10(parallax)
    mu_direct, _ = transformed_distance_quantiles(g0, direct[take])
    mu, distance = transformed_distance_quantiles(g0, calibrated[take])
    distance_truth = 1000 / parallax
    relative = distance[:, 2] / distance_truth - 1
    return {"n": int(take.sum()),
            "direct_Mg": interval_diagnostics(truth, direct[take]),
            "calibrated_Mg": interval_diagnostics(truth, calibrated[take]),
            "direct_final_distance_modulus": interval_diagnostics(mu_truth, mu_direct),
            "final_distance_modulus": interval_diagnostics(mu_truth, mu),
            "relative_distance_bias": float(np.median(relative)),
            "relative_distance_nmad": float(1.4826 * np.median(np.abs(relative - np.median(relative)))),
            "relative_distance_abs_gt_0p5": float(np.mean(np.abs(relative) > .5)),
            "sky_block_ci": sky_block_ci(truth, calibrated[take],
                                         table["ra"][take], table["dec"][take])}


def prepare():
    config, frozen = registration()
    if PREPARED.exists() or (OUT / "route.joblib").exists():
        raise FileExistsError("Cold-v4 preparation already exists")
    references = load_reference_catalogs("data/reference")
    pilot_fit, pilot_test = original_pilot(references)
    sources = [pilot_fit, *(configured_field(path, references) for path in PRIOR_CONFIGS)]
    all_ids = np.concatenate([table["source_id"][take] for table, take, _, _ in sources])
    training_identity = identity_hash(all_ids)
    forests, training, tier_meta = {}, {}, {}
    OUT.mkdir(parents=True, exist_ok=True)
    for tier in FEATURES:
        x, y, ids, counts = per_tier_training(sources, tier)
        model = QuantileForest(n_estimators=300, min_samples_leaf=20).fit(x, y)
        path = OUT / f"{tier}.joblib"
        joblib.dump(model, path)
        forests[tier] = model
        training[tier] = x
        tier_meta[tier] = {"n_fit": len(x), "identity_sha256": identity_hash(ids),
                           "by_field": counts, "model_sha256": sha(path)}
        print(f"fitted {tier}: {len(x)} anchors", flush=True)
    route = fit_route(training, config["selection"]["support_feature_scales"],
                      neighbors=config["selection"]["support_neighbors_other_training"],
                      quantile=config["selection"]["support_training_quantile"])
    route_path = OUT / "route.joblib"
    joblib.dump(route, route_path)
    route_meta = {"optical_low": route["optical_low"].tolist(),
                  "optical_high": route["optical_high"].tolist(),
                  "tiers": {tier: {"fit_rows": item["fit_rows"],
                                   "scale": item["scale"].tolist(),
                                   "radius": item["radius"]}
                            for tier, item in route["tiers"].items()},
                  "route_sha256": sha(route_path)}
    optical_masks = [take & table[LAYER["A_grz"]] &
                     np.isfinite(feature_matrix(table, "A_grz")).all(axis=1)
                     for table, take, _, _ in sources]
    optical_rz = np.concatenate([table["rz0"][mask]
                                 for (table, _, _, _), mask in zip(sources, optical_masks)])
    optical_r = np.concatenate([table["r"][mask]
                                for (table, _, _, _), mask in zip(sources, optical_masks)])
    if not np.isfinite(optical_rz).all() or not np.isfinite(optical_r).all():
        raise ValueError("Nonfinite training colors or depths")
    color_edges = np.r_[np.min(optical_rz) - .01,
                        np.quantile(optical_rz, [1/3, 2/3]),
                        np.max(optical_rz) + .01]
    color_group = np.searchsorted(color_edges[1:-1], optical_rz, side="right")
    r_edges = [float(np.median(optical_r[color_group == i])) for i in range(3)]

    cal_table, cal_cold, cal_mg, cal_meta = configured_field(CAL_CONFIG, references)
    cal_take, cal_direct, cal_route, cal_tiers = routed_prediction(cal_table, cal_cold, route, forests)
    if cal_take.sum() < 300:
        raise ValueError("Too few calibration anchors after complete route")
    groups = group_labels(cal_table["rz0"][cal_take], cal_table["r"][cal_take],
                          color_edges, r_edges)
    offsets = fit_route_offsets(cal_mg[cal_take], cal_direct[cal_take], groups,
                                min_group=80)
    cal_adjusted = cal_direct.copy()
    cal_adjusted[cal_take] = apply_route_offsets(cal_direct[cal_take], groups, offsets)

    common = []
    for table, selected, mg, source in sources:
        matrix = feature_matrix(table, "C_grzW1W2")
        take = (selected & table[LAYER["A_grz"]] & table[LAYER["B_grzW1"]] &
                table[LAYER["C_grzW1W2"]] & np.isfinite(matrix).all(axis=1))
        common.append((table, take, mg, source))
    common_meta = {"n_fit": int(sum(take.sum() for _, take, _, _ in common)),
                   "identity_sha256": identity_hash(np.concatenate(
                       [table["source_id"][take] for table, take, _, _ in common])),
                   "original_frozen_test": {}}
    test_table, test_mask, test_mg, _ = pilot_test
    test_common = (test_mask & test_table[LAYER["A_grz"]] &
                   test_table[LAYER["B_grzW1"]] &
                   test_table[LAYER["C_grzW1W2"]] &
                   np.isfinite(feature_matrix(test_table, "C_grzW1W2")).all(axis=1))
    common_meta["n_test"] = int(test_common.sum())
    for tier in FEATURES:
        x, y, ids, _ = per_tier_training(common, tier)
        if identity_hash(ids) != common_meta["identity_sha256"]:
            raise ValueError(f"Matched-common fit identities differ for {tier}")
        model = QuantileForest(n_estimators=300, min_samples_leaf=20).fit(x, y)
        prediction = model.predict(feature_matrix(test_table, tier)[test_common])
        common_meta["original_frozen_test"][tier] = interval_diagnostics(test_mg[test_common], prediction)
        print(f"matched-common {tier}: {len(x)} fit, {test_common.sum()} test", flush=True)

    prepared = {"registration_sha256": sha(REGISTRATION),
                "frozen_baseline_commit": frozen["baseline_commit"],
                "frozen_baseline_model_sha256": frozen["baseline_model_sha256"],
                "training_identity_sha256": training_identity,
                "training_fields": [source[3] for source in sources],
                "original_frozen_test": pilot_test[3],
                "training_Gaia_BP_RP_counts": cold_color_counts(sources),
                "tier_training": tier_meta, "matched_common_training": common_meta,
                "route": route_meta,
                "route_groups": {"color_edges": color_edges.tolist(),
                                 "r_edges_by_color": r_edges,
                                 "min_group": 80},
                "calibration_field": cal_meta,
                "calibration_selection": {"cold_strict": int(cal_cold.sum()),
                                          "color_box": int((cal_cold & cal_route["color_box"]).sum()),
                                          "selected": int(cal_take.sum()),
                                          "chosen_tier_counts": cal_tiers},
                "calibration_offsets": offsets,
                "calibration_in_sample_metrics": score(cal_table, cal_mg, cal_take,
                                                       cal_direct, cal_adjusted),
                "compute_host": socket.gethostname()}
    write_json(PREPARED, prepared)
    print(json.dumps({"prepared": str(PREPARED),
                      "training_identity": training_identity,
                      "calibration_selected": int(cal_take.sum())}), flush=True)


def final():
    config, frozen = registration()
    if FINAL.exists():
        raise FileExistsError(FINAL)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Published cold-v4 preparation changed")
    prepared = json.loads(PREPARED.read_text())
    if prepared["registration_sha256"] != sha(REGISTRATION):
        raise ValueError("Cold-v4 registration changed")
    if prepared["frozen_baseline_commit"] != frozen["baseline_commit"]:
        raise ValueError("Frozen baseline identity changed")
    route_path = OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Cold-v4 route artifact changed")
    route = joblib.load(route_path)
    forests = {}
    for tier in FEATURES:
        path = OUT / f"{tier}.joblib"
        if sha(path) != prepared["tier_training"][tier]["model_sha256"]:
            raise ValueError(f"Cold-v4 forest changed: {tier}")
        forests[tier] = joblib.load(path)
    edges = np.asarray(prepared["route_groups"]["color_edges"], float)
    r_edges = prepared["route_groups"]["r_edges_by_color"]
    offsets = prepared["calibration_offsets"]
    references = load_reference_catalogs("data/reference")
    summary = {"registration_sha256": sha(REGISTRATION),
               "prepared_sha256": sha(PREPARED),
               "field_roles": {field["name"]: field["role"] for field in config["new_final_fields"]},
               "fields": {}, "compute_host": socket.gethostname()}
    for config_path in FINAL_CONFIGS:
        table, cold, mg, field = configured_field(config_path, references)
        take, direct, routed, counts = routed_prediction(table, cold, route, forests)
        if not take.any():
            summary["fields"][field["name"]] = {"field": field, "selected": 0}
            continue
        groups = group_labels(table["rz0"][take], table["r"][take], edges, r_edges)
        calibrated = direct.copy()
        calibrated[take] = apply_route_offsets(direct[take], groups, offsets)
        result = {"field": field,
                  "selection": {"cold_strict": int(cold.sum()),
                                "color_box": int((cold & routed["color_box"]).sum()),
                                "selected": int(take.sum()),
                                "chosen_tier_counts": counts},
                  "all_selected": score(table, mg, take, direct, calibrated),
                  "application_visible_groups": [],
                  "Gaia_color_diagnostics_only": []}
        selected_indices = np.flatnonzero(take)
        for group in range(6):
            subset = np.zeros(len(take), bool)
            subset[selected_indices[groups == group]] = True
            if subset.any():
                result["application_visible_groups"].append({
                    "group": group, "metrics": score(table, mg, subset, direct, calibrated)})
        color = table["bp_rp_checked"]
        for lo, hi in COLOR_BINS:
            subset = take & (color >= lo) & (color < hi)
            if subset.any():
                result["Gaia_color_diagnostics_only"].append({
                    "BP_RP": [lo, hi], "metrics": score(table, mg, subset, direct, calibrated)})
        with fitsio.FITS(field["path"]) as file:
            mother = file[1].read(columns=sorted(set((*FEATURES["C_grzW1W2"],
                                                     *LAYER.values()))))
        target_route = route_rows(mother, route)
        result["photometric_mother"] = {"all_stored_rows": int(len(mother)),
                                        "grz_base": int(mother["grz_base"].sum()),
                                        "color_box": int(target_route["color_box"].sum()),
                                        "route_selected": int(target_route["selected"].sum()),
                                        "chosen_tier_counts": {tier: int((target_route["tier"] == level).sum())
                                                               for tier, level in LEVEL.items()}}
        summary["fields"][field["name"]] = result
        print(json.dumps({"field": field["name"],
                          "cold_selected": int(take.sum()),
                          "photometric_selected": result["photometric_mother"]["route_selected"],
                          "coverage_95": result["all_selected"]["calibrated_Mg"]["coverage_95"]}), flush=True)
    write_json(FINAL, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "final"))
    args = parser.parse_args()
    require_compute()
    if args.mode == "prepare":
        prepare()
    else:
        final()


if __name__ == "__main__":
    main()

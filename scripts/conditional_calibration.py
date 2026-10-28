#!/usr/bin/env python3
"""Freeze a photometry-only center correction and grouped CQR, then test once."""
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photdis.audit import checked_gaia_color
from photdis.conditional import (FEATURES, apply_group_cqr, correct_center,
                                  fit_center, fit_group_cqr, group_labels, matrix)
from photdis.followup import common_nonoptical, identity_hash
from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.noise import interval_diagnostics, transformed_distance_quantiles
from photdis.stage1 import require_compute, write_json

OUT = Path("data/conditional_v2")
FIG = Path("figures/conditional_v2")
MODEL = OUT / "calibration.joblib"
PREPARED = Path("manifests/conditional_prepared_v2.json")
FINAL = Path("manifests/conditional_final_v2.json")
FOREST_FEATURES = {"A_grz": ("gr0", "rz0"),
                   "B_grzW1": ("gr0", "rz0", "zw10"),
                   "C_grzW1W2": ("gr0", "rz0", "zw10", "w1w20")}
TIERS = {"A_grz": "grz_base", "B_grzW1": "grz_w1", "C_grzW1W2": "grz_w1_w2"}
GAIA = ["gaia_gaia_match", "gaia_phot_bp_mean_mag", "gaia_phot_rp_mean_mag",
        "gaia_phot_bp_mean_flux", "gaia_phot_rp_mean_flux",
        "gaia_phot_bp_mean_flux_error", "gaia_phot_rp_mean_flux_error",
        "gaia_phot_bp_n_obs", "gaia_phot_rp_n_obs", "gaia_bp_rp"]
READ_COLUMNS = sorted(set(["source_id", "ra", "dec", "g0", "p_corr", "anchor",
                           "anchor_astrometry", "contaminant_candidate",
                           "desi_quality", "desi_known_contaminant", "desi_match_ambiguous",
                           "desi_logg", "apogee_quality", "apogee_known_contaminant",
                           "apogee_match_ambiguous", "apogee_logg",
                           *TIERS.values(), *GAIA,
                           *{name for features in FEATURES.values() for name in features}]))
CONFIGS = {
    "prior_cal": "configs/field_cal_ra140_150.json",
    "prior_north": "configs/field_test_ra190_200.json",
    "prior_south": "configs/field_test_ra320_330.json",
    "new_cal": "configs/field_dev_cal_ra205_215.json",
    "final_north": "configs/field_final_ra230_240.json",
    "final_south": "configs/field_final_ra050_060.json",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_field(name, references):
    task = json.loads(Path(CONFIGS[name]).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    with fitsio.FITS(str(path)) as file:
        rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        raw = file[1].read(columns=READ_COLUMNS, rows=rows)
    table = {key: raw[key] for key in raw.dtype.names}
    gaia = {key[5:]: table[key] for key in GAIA}
    table.update(checked_gaia_color(gaia))
    external = annotate_holdouts(references, table["source_id"],
                                 table["ra"], table["dec"])["external_any"]
    eligible, _, mg = common_nonoptical(table, external)
    selected = table["anchor"] & eligible
    return table, selected, mg, {"name": task["name"],
                                 "bounds_deg": task["bounds_deg"],
                                 "all_strict_anchor_rows": len(raw),
                                 "cold_strict_anchor_rows": int(selected.sum()),
                                 "cold_identity_sha256": identity_hash(table["source_id"][selected])}


def tier_rows(field, tier, forest):
    table, selected, mg, _ = field
    mask = selected & table[TIERS[tier]]
    x_forest = np.column_stack([table[key][mask] for key in FOREST_FEATURES[tier]])
    x_center = matrix({key: table[key][mask] for key in FEATURES[tier]}, tier)
    good = np.all(np.isfinite(x_forest), axis=1) & np.all(np.isfinite(x_center), axis=1)
    indices = np.flatnonzero(mask)[good]
    if len(indices) == 0:
        raise ValueError(f"No valid {tier} cold stars in field")
    prediction = forest.predict(x_forest[good])
    return {"truth": np.asarray(mg[indices], float),
            "prediction": prediction,
            "center_features": x_center[good],
            "rz0": np.asarray(table["rz0"][indices], float),
            "r": np.asarray(table["r"][indices], float),
            "g0": np.asarray(table["g0"][indices], float),
            "p_corr": np.asarray(table["p_corr"][indices], float),
            "bp_rp": np.asarray(table["bp_rp_checked"][indices], float),
            "ra": np.asarray(table["ra"][indices], float),
            "dec": np.asarray(table["dec"][indices], float),
            "source_id": table["source_id"][indices]}


def concatenate(rows, key):
    return np.concatenate([item[key] for item in rows])


def sky_block_ci(truth, prediction, ra, dec, n_boot=300, seed=20261028):
    cells = hp.ang2pix(64, ra, dec, nest=True, lonlat=True)
    groups = np.unique(cells)
    if len(groups) < 5:
        return None
    lookup = [np.flatnonzero(cells == cell) for cell in groups]
    rng = np.random.default_rng(seed)
    coverage, nmad = [], []
    for _ in range(n_boot):
        take = np.concatenate([lookup[i] for i in rng.integers(len(groups), size=len(groups))])
        result = interval_diagnostics(truth[take], prediction[take])
        coverage.append(result["coverage_95"])
        nmad.append(result["nmad"])
    return {"n_nside64_blocks": len(groups),
            "coverage_95_95pct": np.quantile(coverage, [.025, .975]).tolist(),
            "nmad_95pct": np.quantile(nmad, [.025, .975]).tolist()}


def evaluate(rows, corrected, calibrated, edges, r_edges):
    result = {"n": len(rows["truth"]),
              "direct_Mg": interval_diagnostics(rows["truth"], rows["prediction"]),
              "center_Mg": interval_diagnostics(rows["truth"], corrected),
              "center_and_group_cqr_Mg": interval_diagnostics(rows["truth"], calibrated)}
    true_mu = 10 - 5 * np.log10(rows["p_corr"])
    mu, distance = transformed_distance_quantiles(rows["g0"], calibrated)
    result["final_distance_modulus"] = interval_diagnostics(true_mu, mu)
    true_distance = 1000 / rows["p_corr"]
    result["final_distance_relative_nmad"] = float(1.4826 * np.median(np.abs(
        distance[:, 2] / true_distance - 1 - np.median(distance[:, 2] / true_distance - 1))))
    result["sky_block_ci"] = sky_block_ci(rows["truth"], calibrated,
                                           rows["ra"], rows["dec"])
    groups = group_labels(rows["rz0"], rows["r"], edges, r_edges)
    result["application_visible_bins"] = []
    for group in range(6):
        take = groups == group
        if take.sum() == 0:
            continue
        item = {"group": group, "n": int(take.sum()),
                "r_z_bin": group // 2, "r_depth_bin": group % 2,
                "metrics": interval_diagnostics(rows["truth"][take], calibrated[take])}
        if take.sum() >= 20:
            item["sky_block_ci"] = sky_block_ci(rows["truth"][take], calibrated[take],
                                                 rows["ra"][take], rows["dec"][take])
        result["application_visible_bins"].append(item)
    result["Gaia_color_diagnostics_only"] = []
    for lo, hi in zip((1.3, 1.8, 2.3, 2.8), (1.8, 2.3, 2.8, 3.5)):
        take = (rows["bp_rp"] >= lo) & (rows["bp_rp"] < hi)
        if take.sum():
            result["Gaia_color_diagnostics_only"].append({"BP_RP": [lo, hi],
                "direct": interval_diagnostics(rows["truth"][take], rows["prediction"][take]),
                "center": interval_diagnostics(rows["truth"][take], corrected[take]),
                "center_and_group_cqr": interval_diagnostics(rows["truth"][take], calibrated[take])})
    return result


def prepare(forests, references, frozen):
    if PREPARED.exists() or MODEL.exists():
        raise FileExistsError("Conditional preparation already exists")
    prior = [load_field(name, references) for name in ("prior_north", "prior_south")]
    diagnostic = load_field("prior_cal", references)
    calibration = load_field("new_cal", references)
    manifest = {"method": "Ridge(alpha=100) on clipped predicted-minus-Gaia Mg residual; three development-rz quantile bins, each split at its own development-r median; split-CQR minimum 80 per group and fixed fallback by unlabeled count",
                "photometry_only_center_features": FEATURES,
                "development_fields": [item[3] for item in prior],
                "development_diagnostic_field": diagnostic[3],
                "independent_calibration_field": calibration[3],
                "frozen_baseline_commit": frozen["baseline_commit"],
                "tiers": {}, "compute_host": socket.gethostname()}
    models = {}
    for tier, forest in forests.items():
        train = [tier_rows(field, tier, forest) for field in prior]
        truth = concatenate(train, "truth")
        prediction = concatenate(train, "prediction")
        x = concatenate(train, "center_features")
        scaler, center = fit_center(x, prediction[:, 2] - truth)
        training_rz = concatenate(train, "rz0")
        edges = [float(np.min(training_rz) - .1),
                 *np.quantile(training_rz, [1/3, 2/3]).tolist(),
                 float(np.max(training_rz) + .1)]
        training_r = concatenate(train, "r")
        color_index = np.searchsorted(edges[1:-1], training_rz, side="right")
        r_edges = [float(np.median(training_r[color_index == group]))
                   for group in range(3)]
        cal = tier_rows(calibration, tier, forest)
        corrected = correct_center(cal["prediction"], cal["center_features"], scaler, center)
        groups = group_labels(cal["rz0"], cal["r"], edges, r_edges)
        offsets = fit_group_cqr(cal["truth"], corrected, groups, min_group=80)
        adjusted = apply_group_cqr(corrected, groups, offsets)
        dev_check = tier_rows(diagnostic, tier, forest)
        dev_corrected = correct_center(dev_check["prediction"],
                                       dev_check["center_features"], scaler, center)
        models[tier] = {"scaler": scaler, "center": center,
                        "color_edges": edges, "r_edges_by_color": r_edges,
                        "offsets": offsets}
        manifest["tiers"][tier] = {
            "development_n": len(truth),
            "development_identity_sha256": identity_hash(concatenate(train, "source_id")),
            "calibration_n": len(cal["truth"]),
            "calibration_identity_sha256": identity_hash(cal["source_id"]),
            "color_edges_rz0": edges, "r_edges_by_color": r_edges,
            "offsets": offsets,
            "prior_cal_development_check": {
                "direct": interval_diagnostics(dev_check["truth"], dev_check["prediction"]),
                "center": interval_diagnostics(dev_check["truth"], dev_corrected)},
            "new_calibration_apparent": evaluate(cal, corrected, adjusted, edges, r_edges)}
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL)
    manifest["model_sha256"] = sha(MODEL)
    write_json(OUT / "prepared_summary.json", manifest)
    write_json(PREPARED, manifest)
    print(json.dumps({"prepared": {tier: {"dev": item["development_n"],
                                      "cal": item["calibration_n"]}
                                   for tier, item in manifest["tiers"].items()}}), flush=True)


def final_test(forests, references, frozen):
    if FINAL.exists() or (OUT / "final_summary.json").exists():
        raise FileExistsError("Final evaluation already exists")
    prepared = json.loads(PREPARED.read_text())
    if prepared["frozen_baseline_commit"] != frozen["baseline_commit"] or sha(MODEL) != prepared["model_sha256"]:
        raise ValueError("Prepared calibration artifact changed")
    registered = json.loads(Path("configs/next_fields.json").read_text())
    models = joblib.load(MODEL)
    summary = {"method_manifest_sha256": sha(PREPARED),
               "field_roles": {item["name"]: item["role"] for item in registered["fields"]},
               "fields": {}, "compute_host": socket.gethostname()}
    for key in ("final_north", "final_south"):
        field = load_field(key, references)
        name = field[3]["name"]
        if summary["field_roles"].get(name) != "untouched_final_test":
            raise ValueError("Final field role was not preregistered")
        item = {"field": field[3], "tiers": {}}
        cache = {}
        for tier, forest in forests.items():
            rows = tier_rows(field, tier, forest)
            model = models[tier]
            corrected = correct_center(rows["prediction"], rows["center_features"],
                                       model["scaler"], model["center"])
            groups = group_labels(rows["rz0"], rows["r"],
                                  model["color_edges"], model["r_edges_by_color"])
            adjusted = apply_group_cqr(corrected, groups, model["offsets"])
            item["tiers"][tier] = evaluate(rows, corrected, adjusted,
                                             model["color_edges"], model["r_edges_by_color"])
            cache[tier] = (rows, adjusted)
        common_ids = set(cache["C_grzW1W2"][0]["source_id"].tolist())
        for tier in FOREST_FEATURES:
            common_ids.intersection_update(cache[tier][0]["source_id"].tolist())
        common_ids = np.array(sorted(common_ids), dtype=np.int64)
        item["same_star_ABC"] = {"n": len(common_ids),
                                  "identity_sha256": identity_hash(common_ids),
                                  "tiers": {}}
        for tier, (rows, adjusted) in cache.items():
            row_lookup = {int(sid): i for i, sid in enumerate(rows["source_id"])}
            take = np.array([row_lookup[int(sid)] for sid in common_ids], dtype=int)
            item["same_star_ABC"]["tiers"][tier] = interval_diagnostics(
                rows["truth"][take], adjusted[take])
        summary["fields"][name] = item
        print(json.dumps({"final_field": name,
                          "tiers": {tier: value["center_and_group_cqr_Mg"]
                                    for tier, value in item["tiers"].items()}}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for tier, color in zip(FOREST_FEATURES, ("#356e9b", "#58894e", "#bf6f38")):
        labels = list(summary["fields"])
        x = np.arange(len(labels))
        direct = [summary["fields"][field]["tiers"][tier]["direct_Mg"]["coverage_95"]
                  for field in labels]
        calibrated = [summary["fields"][field]["tiers"][tier]
                      ["center_and_group_cqr_Mg"]["coverage_95"] for field in labels]
        bias = [summary["fields"][field]["tiers"][tier]["center_Mg"]["bias"]
                for field in labels]
        axes[0].plot(x, bias, "o-", label=tier, color=color)
        axes[1].plot(x, direct, "o--", color=color, alpha=.55)
        axes[1].plot(x, calibrated, "o-", color=color, label=tier)
    axes[0].axhline(0, color=".5", ls=":")
    axes[1].axhline(.95, color=".5", ls=":")
    for ax in axes:
        ax.set_xticks(np.arange(len(labels)), ["RA 230-240", "RA 050-060"])
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Centered Mg bias [mag]")
    axes[1].set_ylabel("95% Mg coverage (solid calibrated)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_final_conditional_calibration.png", dpi=170)
    plt.close(fig)
    write_json(OUT / "final_summary.json", summary)
    write_json(FINAL, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "final"))
    args = parser.parse_args()
    require_compute()
    frozen = json.loads(Path("configs/followup_fields.json").read_text())
    if sha("manifests/cold_baseline.json") != frozen["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline manifest changed")
    forests = {}
    for tier, digest in frozen["baseline_model_sha256"].items():
        path = Path("data/cold_baseline_v3") / f"{tier}.joblib"
        if sha(path) != digest:
            raise ValueError(f"Frozen forest changed: {tier}")
        forests[tier] = joblib.load(path)
    references = load_reference_catalogs("data/reference")
    if args.mode == "prepare":
        prepare(forests, references, frozen)
    else:
        final_test(forests, references, frozen)


if __name__ == "__main__":
    main()

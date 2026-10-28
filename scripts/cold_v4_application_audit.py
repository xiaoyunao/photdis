#!/usr/bin/env python3
"""Audit frozen cold-v4 routing by tier, image quality, error and support."""
from __future__ import annotations

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
from photdis.route import FEATURES, LAYER, LEVEL, apply_route_offsets, feature_matrix, route_rows
from photdis.stage1 import require_compute, write_json
from cold_v4 import (CAL_CONFIG, FINAL_CONFIGS, OUT, PREPARED,
                     PREPARED_TRACKED, registration, sha)
from conditional_calibration import GAIA, READ_COLUMNS

RESULT = Path("data/cold_v4_application/audit.json")
QUALITY = ("clean_precise_optical", "clean_loose_optical", "base_nonclean")
ERROR_BINS = ((0, .03), (.03, .10), (.10, .20), (.20, float("inf")))
SUPPORT_BINS = ((0, .5), (.5, .8), (.8, 1.000001))
EXTRA = ("gaia_with_grz", "grz_clean", "grz_high_quality", "grz_fraction_clean",
         "grz_mask_clean", "grz_w1_high_quality", "full_high_quality",
         "sigma_g", "sigma_r", "sigma_z", "sigma_w1", "sigma_w2",
         "snr_w1", "snr_w2", "r")
MOTHER_COLS = sorted(set((*FEATURES["C_grzW1W2"], *LAYER.values(), *EXTRA,
                          "source_id")))
GAIA_COLS = sorted(set((*READ_COLUMNS, *EXTRA, "gaia_phot_g_mean_mag")))


def quality_group(table):
    clean = np.asarray(table["grz_clean"], bool)
    precise = np.asarray(table["grz_high_quality"], bool)
    return {QUALITY[0]: precise,
            QUALITY[1]: clean & ~precise,
            QUALITY[2]: ~clean}


def tier_error_and_support(table, routed, route):
    """Use only bands and the fitted support space of the chosen tier."""
    n = len(routed["tier"])
    sigma = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    for tier, level in LEVEL.items():
        selected = routed["tier"] == level
        if not selected.any():
            continue
        used_bands = ("g", "r", "z") + (("w1",) if level >= 2 else ()) + (("w2",) if level == 3 else ())
        sigma[selected] = np.maximum.reduce(
            [np.asarray(table[f"sigma_{band}"][selected], float) for band in used_bands])
        model = route["tiers"][tier]
        colors = feature_matrix(table, tier)[selected] / model["scale"]
        distance = model["tree"].query(colors, k=route["neighbors"])[0][:, -1]
        ratio[selected] = distance / model["radius"]
    if not np.isfinite(sigma[routed["selected"]]).all() or not np.isfinite(ratio[routed["selected"]]).all():
        raise ValueError("Selected target has nonfinite used-band error or support position")
    return sigma, ratio


def tier_quality(table, tier):
    name = {"A_grz": "grz_high_quality",
            "B_grzW1": "grz_w1_high_quality",
            "C_grzW1W2": "full_high_quality"}[tier]
    return np.asarray(table[name], bool)


def count_groups(table, routed, sigma, ratio):
    selected = routed["selected"]
    q = quality_group(table)
    result = {"all_stored_rows": len(selected),
              "grz_base_rows": int(np.sum(table["grz_base"])),
              "grz_clean_rows": int(np.sum(table["grz_clean"])),
              "color_box_rows": int(np.sum(routed["color_box"])),
              "route_selected_rows": int(selected.sum()),
              "route_selected_and_clean": int(np.sum(selected & table["grz_clean"])),
              "route_selected_and_high_quality_optical": int(np.sum(selected & table["grz_high_quality"])),
              "fraction_clean_failure": int(np.sum(selected & ~table["grz_fraction_clean"])),
              "mask_clean_failure": int(np.sum(selected & ~table["grz_mask_clean"])),
              "tier_x_quality": {}, "tier_x_used_error": {},
              "tier_x_support_ratio": {}, "tier_high_quality": {},
              "fallback_after_unsupported_higher_layer": {}}
    for tier, level in LEVEL.items():
        take = selected & (routed["tier"] == level)
        result["tier_x_quality"][tier] = {name: int(np.sum(take & mask)) for name, mask in q.items()}
        result["tier_high_quality"][tier] = int(np.sum(take & tier_quality(table, tier)))
        result["tier_x_used_error"][tier] = {
            f"{lo:g}_{hi:g}": int(np.sum(take & (sigma >= lo) & (sigma < hi)))
            for lo, hi in ERROR_BINS}
        result["tier_x_support_ratio"][tier] = {
            f"{lo:g}_{hi:g}": int(np.sum(take & (ratio >= lo) & (ratio < hi)))
            for lo, hi in SUPPORT_BINS}
        higher = [name for name in FEATURES if LEVEL[name] > level]
        result["fallback_after_unsupported_higher_layer"][tier] = int(np.sum(
            take & np.logical_or.reduce([
                np.asarray(table[LAYER[name]], bool) & ~routed["support_by_tier"][name]
                for name in higher]) if higher else np.zeros(len(take), bool)))
    assert sum(sum(v.values()) for v in result["tier_x_quality"].values()) == result["route_selected_rows"]
    return result


def block_intervals(truth, prediction, ra, dec, seed=20261028, repeats=250):
    if len(truth) < 30:
        return None
    cell = hp.ang2pix(64, ra, dec, nest=True, lonlat=True)
    unique, inverse = np.unique(cell, return_inverse=True)
    if len(unique) < 5:
        return None
    lookup = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    values = np.empty((repeats, 3))
    for iteration in range(repeats):
        take = np.concatenate([lookup[i] for i in rng.integers(len(unique), size=len(unique))])
        metric = interval_diagnostics(truth[take], prediction[take])
        values[iteration] = (metric["bias"], metric["nmad"], metric["coverage_95"])
    return {"n_nside64_blocks": len(unique),
            "bias_95pct": np.quantile(values[:, 0], [.025, .975]).tolist(),
            "nmad_95pct": np.quantile(values[:, 1], [.025, .975]).tolist(),
            "coverage_95_95pct": np.quantile(values[:, 2], [.025, .975]).tolist()}


def prediction_metrics(table, selected, direct, calibrated):
    n = int(selected.sum())
    if not n:
        return {"n": 0}
    g0 = np.asarray(table["g0"][selected], float)
    parallax = np.asarray(table["p_corr"][selected], float)
    truth_mu = 10 - 5 * np.log10(parallax)
    direct_mu, _ = transformed_distance_quantiles(g0, direct[selected])
    mu, distance = transformed_distance_quantiles(g0, calibrated[selected])
    relative = distance[:, 2] * parallax / 1000 - 1
    return {"n": n,
            "direct_final_distance_modulus": interval_diagnostics(truth_mu, direct_mu),
            "final_distance_modulus": interval_diagnostics(truth_mu, mu),
            "relative_distance_bias": float(np.median(relative)),
            "relative_distance_nmad": float(1.4826 * np.median(np.abs(relative - np.median(relative)))),
            "relative_distance_abs_gt_0p5": float(np.mean(np.abs(relative) > .5)),
            "sky_block_uncertainty": block_intervals(truth_mu, mu,
                                                     table["ra"][selected], table["dec"][selected])}


def load_gaia_and_mother(path, references):
    with fitsio.FITS(str(path)) as file:
        hdu = file[1]
        rows = np.flatnonzero(hdu.read(columns=["gaia_with_grz"])["gaia_with_grz"])
        raw = hdu.read(columns=GAIA_COLS, rows=rows)
        mother = hdu.read(columns=MOTHER_COLS)
    table = {name: raw[name] for name in raw.dtype.names}
    ids = table["source_id"][table["source_id"] > 0]
    if len(ids) != len(np.unique(ids)):
        raise ValueError(f"Duplicate Gaia source_id among reference rows: {path}")
    gaia = {name[5:]: table[name] for name in GAIA}
    table.update(checked_gaia_color(gaia))
    external = annotate_holdouts(references, table["source_id"],
                                 table["ra"], table["dec"])["external_any"]
    with np.errstate(divide="ignore", invalid="ignore"):
        legacy_cold, flags, _ = common_nonoptical(table, external)
        gaia_absolute_g = (table["gaia_phot_g_mean_mag"] - 10 +
                           5 * np.log10(table["p_corr"]))
    # This diagnostic branch does not use potentially spoiled Legacy g
    # to decide whether a nonclean row is a dwarf.
    independent = [flag for name, flag in flags.items()
                   if name != "cold_CMD_luminosity"]
    cold = (np.logical_and.reduce(independent) & np.isfinite(gaia_absolute_g) &
            (gaia_absolute_g > 5) & np.isfinite(table["p_corr"]) &
            (table["p_corr"] > 0))
    return table, cold, legacy_cold, {name: mother[name] for name in mother.dtype.names}


def summarize_field(config_path, references, route, forests, prepared):
    task = json.loads(Path(config_path).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    table, cold, legacy_cold, mother = load_gaia_and_mother(path, references)
    routed = route_rows(table, route)
    selected = cold & routed["selected"]
    if not selected.any():
        raise ValueError(f"No cold Gaia reference selected in {task['name']}")
    sigma, ratio = tier_error_and_support(table, routed, route)
    mother_route = route_rows(mother, route)
    mother_sigma, mother_ratio = tier_error_and_support(mother, mother_route, route)
    groups = group_labels(table["rz0"][selected], table["r"][selected],
                          prepared["route_groups"]["color_edges"],
                          prepared["route_groups"]["r_edges_by_color"])
    direct = np.full((len(cold), 5), np.nan)
    for tier, level in LEVEL.items():
        mask = selected & (routed["tier"] == level)
        if mask.any():
            direct[mask] = forests[tier].predict(feature_matrix(table, tier)[mask])
    calibrated = direct.copy()
    calibrated[selected] = apply_route_offsets(direct[selected], groups,
                                               prepared["calibration_offsets"])
    quality = quality_group(table)
    result = {"field": task["name"], "path": str(path),
              "gaia_with_grz_rows": len(cold),
              "cold_Gaia_base_rows": int(cold.sum()),
              "cold_Gaia_identity_sha256": identity_hash(table["source_id"][cold]),
              "cold_Gaia_strict_anchor_rows": int(np.sum(cold & table["anchor"])),
              "legacy_Mg_branch_strict_anchor_rows": int(np.sum(legacy_cold & table["anchor"])),
              "cold_Gaia_route_selected": int(selected.sum()),
              "photometric_candidate_distribution": count_groups(mother, mother_route,
                                                                   mother_sigma, mother_ratio),
              "cold_Gaia_distribution": count_groups(table, routed, sigma, ratio),
              "cold_Gaia_metrics": {}}
    strata = {"all": selected, "strict_anchor": selected & table["anchor"],
              "legacy_Mg_branch_strict_anchor": (selected & legacy_cold & table["anchor"])}
    strata.update({name: selected & mask for name, mask in quality.items()})
    for tier, level in LEVEL.items():
        take = selected & (routed["tier"] == level)
        strata[tier] = take
        strata[tier + "_strict_anchor"] = take & table["anchor"]
        strata[tier + "_base_nonclean"] = take & quality[QUALITY[2]]
        strata[tier + "_clean_loose"] = take & quality[QUALITY[1]]
        strata[tier + "_clean_precise"] = take & quality[QUALITY[0]]
        for lo, hi in ERROR_BINS:
            strata[f"{tier}_used_error_{lo:g}_{hi:g}"] = take & (sigma >= lo) & (sigma < hi)
        for lo, hi in SUPPORT_BINS:
            strata[f"{tier}_support_ratio_{lo:g}_{hi:g}"] = take & (ratio >= lo) & (ratio < hi)
    for name, mask in strata.items():
        if mask.any():
            result["cold_Gaia_metrics"][name] = prediction_metrics(table, mask, direct, calibrated)
    print(json.dumps({"field": task["name"],
                      "gaia_cold": result["cold_Gaia_base_rows"],
                      "gaia_selected": result["cold_Gaia_route_selected"],
                      "photometric_selected": result["photometric_candidate_distribution"]["route_selected_rows"]}),
          flush=True)
    return result


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    registration()
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Published cold-v4 preparation changed")
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
    references = load_reference_catalogs("data/reference")
    summary = {"prepared_sha256": sha(PREPARED),
               "registration_sha256": sha("configs/cold_v4_fields.json"),
               "fields": {}, "compute_host": socket.gethostname(),
               "metric_scope": "Gaia high-quality astrometry, BP-RP 1.3–3.5 and absolute Gaia G >5 with grz_base; photometric candidates have no distance truth. The old Legacy Mg>5 strict branch is separately counted.",
               "quality_group_definition": {QUALITY[0]: "grz_high_quality",
                                            QUALITY[1]: "grz_clean and not grz_high_quality",
                                            QUALITY[2]: "grz_base and not grz_clean"},
               "error_definition": "largest formal magnitude error among bands used by selected tier",
               "support_definition": "distance to tenth training neighbor divided by tier-specific 95th-percentile support radius"}
    for config in (CAL_CONFIG, *FINAL_CONFIGS):
        field = summarize_field(config, references, route, forests, prepared)
        summary["fields"][field["field"]] = field
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, summary)


if __name__ == "__main__":
    main()

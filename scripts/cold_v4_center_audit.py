#!/usr/bin/env python3
"""Diagnose frozen-v4 final-field modulus offsets without fitting a correction."""
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

from photdis.audit import rvs_metallicity
from photdis.conditional import group_labels
from photdis.noise import interval_diagnostics, transformed_distance_quantiles
from photdis.route import FEATURES, LEVEL, apply_route_offsets, feature_matrix
from photdis.stage1 import require_compute, write_json
from cold_v4 import (FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED,
                     configured_field, routed_prediction, sha)

RESULT = Path("data/cold_v4_center/diagnostic.json")
EXTRA = ["source_id", "ebv", "r", "sigma_g", "sigma_r", "sigma_z",
         "sigma_w1", "sigma_w2", "gaia_parallax_error", "gaia_ruwe",
         "gaia_non_single_star", "gaia_ipd_frac_multi_peak",
         "desi_quality", "desi_is_stellar", "desi_feh", "desi_feh_err",
         "desi_teff", "desi_logg"]


def group_score(mask, truth, quantiles):
    n = int(np.sum(mask))
    if n < 20:
        return {"n": n, "status": "below_20"}
    return interval_diagnostics(truth[mask], quantiles[mask])


def bins(values, edges, truth, quantiles):
    values = np.asarray(values, float)
    return [{"bin": [lo, hi], **group_score((values >= lo) & (values < hi), truth, quantiles)}
            for lo, hi in zip(edges[:-1], edges[1:])]


def field_result(config_path, route, forests, prepared, references):
    task = json.loads(Path(config_path).read_text())
    table, cold, _, metadata = configured_field(config_path, references)
    with fitsio.FITS(metadata["path"]) as file:
        rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        extra = file[1].read(columns=EXTRA, rows=rows)
    if not np.array_equal(extra["source_id"], table["source_id"]):
        raise ValueError("Center sidecar does not match anchor rows")
    take, direct, routed, counts = routed_prediction(table, cold, route, forests)
    selected_rows = np.flatnonzero(take)
    if not len(selected_rows):
        return {"field": task["name"], "n": 0}
    color_edges = np.asarray(prepared["route_groups"]["color_edges"], float)
    groups = group_labels(table["rz0"][take], extra["r"][take], color_edges,
                          prepared["route_groups"]["r_edges_by_color"])
    calibrated = apply_route_offsets(direct[take], groups,
                                     prepared["calibration_offsets"])
    truth = 10 - 5 * np.log10(np.asarray(table["p_corr"][take], float))
    mu = transformed_distance_quantiles(np.asarray(table["g0"][take], float),
                                        calibrated)[0]
    chosen_tier = routed["tier"][take]
    sigma = np.full(len(selected_rows), np.nan)
    support = np.full(len(selected_rows), np.nan)
    for tier, level in LEVEL.items():
        use = chosen_tier == level
        if not use.any():
            continue
        bands = ("g", "r", "z") + (("w1",) if level >= 2 else ()) + (("w2",) if level == 3 else ())
        sigma[use] = np.maximum.reduce([np.asarray(extra[f"sigma_{band}"][take][use], float)
                                        for band in bands])
        model = route["tiers"][tier]
        distance = model["tree"].query(feature_matrix(table, tier)[take][use] /
                                       model["scale"], k=route["neighbors"])[0][:, -1]
        support[use] = distance / model["radius"]
    teff = np.asarray(extra["desi_teff"][take], float)
    logg = np.asarray(extra["desi_logg"][take], float)
    feh, valid = rvs_metallicity(extra["desi_feh"][take], teff, logg)
    valid &= (extra["desi_quality"][take] & extra["desi_is_stellar"][take] &
              np.isfinite(extra["desi_feh_err"][take]) &
              (extra["desi_feh_err"][take] > 0))
    snr = np.asarray(table["p_corr"][take], float) / np.asarray(extra["gaia_parallax_error"][take], float)
    non_single = np.asarray(extra["gaia_non_single_star"][take], bool)
    result = {"field": task["name"], "selected_strict_cold": len(selected_rows),
              "selected_tiers": counts,
              "overall": interval_diagnostics(truth, mu),
              "by_original_route_group": {str(i): group_score(groups == i, truth, mu)
                                          for i in range(6)},
              "by_tier": {tier: group_score(chosen_tier == level, truth, mu)
                          for tier, level in LEVEL.items()},
              "by_r": bins(extra["r"][take], [14, 17, 19, 21, 24], truth, mu),
              "by_r_minus_z": bins(table["rz0"][take], [0, .6, 1., 1.6, 3.5], truth, mu),
              "by_used_band_sigma": bins(sigma, [0, .03, .1, .2, 2], truth, mu),
              "by_support_radius_ratio": bins(support, [0, .5, .8, 1.01], truth, mu),
              "by_corrected_parallax_snr": bins(snr, [10, 15, 25, 10000], truth, mu),
              "by_ebv": bins(extra["ebv"][take], [0, .01, .02, .03, 1], truth, mu),
              "by_gaia_ruwe": bins(extra["gaia_ruwe"][take], [0, 1.1, 1.25, 1.4, 10], truth, mu),
              "by_gaia_non_single_star": {
                  "false": group_score(~non_single, truth, mu),
                  "true": group_score(non_single, truth, mu)},
              "by_gaia_ipd_multi_peak": {
                  "zero": group_score(extra["gaia_ipd_frac_multi_peak"][take] == 0, truth, mu),
                  "positive": group_score(extra["gaia_ipd_frac_multi_peak"][take] > 0, truth, mu)},
              "calibrated_desi_metallicity_valid": int(valid.sum()),
              "by_calibrated_desi_feh": {
                  "below_minus_1p5": group_score(valid & (feh < -1.5), truth, mu),
                  "minus_1p5_to_minus_0p5": group_score(valid & (feh >= -1.5) & (feh < -.5), truth, mu),
                  "above_minus_0p5": group_score(valid & (feh >= -.5), truth, mu)},
              "sky_quadrants": {}}
    ra0, dec0, ra1, dec1 = task["bounds_deg"]
    for i in range(2):
        for j in range(2):
            sky = ((table["ra"][take] >= (ra0 + ra1) / 2 if i else table["ra"][take] < (ra0 + ra1) / 2) &
                   (table["dec"][take] >= (dec0 + dec1) / 2 if j else table["dec"][take] < (dec0 + dec1) / 2))
            result["sky_quadrants"][f"ra{i}_dec{j}"] = group_score(sky, truth, mu)
    return result


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen v4 preparation changed")
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
    from photdis.holdouts import load_reference_catalogs
    references = load_reference_catalogs("data/reference")
    result = {"prepared_sha256": sha(PREPARED), "compute_host": socket.gethostname(),
              "scope": "univariate diagnostics of frozen strict-Gaia cold anchors; no center fit",
              "fields": {}}
    for config in FINAL_CONFIGS:
        row = field_result(config, route, forests, prepared, references)
        result["fields"][row["field"]] = row
        print(json.dumps({"field": row["field"], "n": row["selected_strict_cold"],
                          "bias": row["overall"]["bias"]}), flush=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bounded grz-only luminosity mixture for warm spectroscopic giant evidence."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np
from scipy.spatial import Delaunay

from photdis.giant_kernel import (color_covariance, conditional_weights,
                                  mixture_logpdf, predict_one)
from photdis.stage1 import require_compute, write_json

SOURCE = Path("data/giant_vac_v3/same_star.fits")
M3 = Path("manifests/giant_m3_reference_v1.json")
M3_FIELD = Path("data/field_sky_ra200_210_v1/field_diagnostics.fits")
OUT = Path("manifests/giant_photo_baseline_v1.json")
FIELD_ROLES = ("development", "reserved_validation")
BANDWIDTHS = (.06, .12, .24)
MAG_FLOORS = (.15, .30, .60)


def features(rows):
    color = np.column_stack((rows["gr0"], rows["rz0"])).astype(float)
    covariance = color_covariance(rows["sigma_g"], rows["sigma_r"],
                                  rows["sigma_z"])
    return color, covariance


def field_name(value):
    """FITS readers may expose the same string column as bytes or Unicode."""
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()


def clean_warm_spdist(rows):
    return (rows["clean_low_dust"] & ~rows["cmd_spectral_conflict"]
            & rows["spdist_valid"] & (rows["cluster_group"] == 0)
            & (rows["desi_teff"] >= 5000) & (rows["desi_teff"] < 7000)
            & np.isfinite(rows["gr0"]) & np.isfinite(rows["rz0"])
            & np.isfinite(rows["sigma_r"]) & np.isfinite(rows["sigma_z"])
            & (rows["sigma_g"] > 0) & (rows["sigma_r"] > 0)
            & (rows["sigma_z"] > 0) & (rows["spdist_sigma_mg"] > 0))


def density_score(train, queries, bandwidth, mag_floor):
    x, cov = features(train)
    qx, qcov = features(queries)
    centers = train["spdist_mg_legacy"].astype(float)
    width = np.hypot(train["spdist_sigma_mg"].astype(float), mag_floor)
    ll = []
    for i, row in enumerate(queries):
        weights, _ = conditional_weights(x, cov, qx[i], qcov[i], bandwidth)
        ll.append(mixture_logpdf(row["spdist_mg_legacy"], centers, width,
                                 weights, row["spdist_sigma_mg"]))
    return float(np.mean(ll))


def estimate(rows, training, bandwidth, mag_floor):
    x, cov = features(training)
    qx, qcov = features(rows)
    inside = Delaunay(x).find_simplex(qx) >= 0
    means = training["spdist_mg_legacy"].astype(float)
    errors = training["spdist_sigma_mg"].astype(float)
    values = []
    for i, row in enumerate(rows):
        pred = predict_one(x, cov, means, errors, qx[i], qcov[i],
                           bandwidth, mag_floor,
                           observed_mg=row["spdist_mg_legacy"],
                           observed_sigma_mg=row["spdist_sigma_mg"])
        quantiles = pred["quantiles"]
        values.append((int(row["source_id"]), field_name(row["field"]),
                       row["spdist_mg_legacy"].item(),
                       row["spdist_sigma_mg"].item(),
                       row["specdis_mg_legacy"].item() if row["specdis_valid"] else None,
                       row["specdis_sigma_mg"].item() if row["specdis_valid"] else None,
                       *quantiles.tolist(), pred["effective_n"],
                       pred["log_density"],
                       bool(inside[i] and pred["effective_n"] >= 20)))
    return values


def summarize(values):
    if not values:
        return {"n": 0}
    records = np.asarray(values, object)
    true = records[:, 2].astype(float)
    pred = records[:, 8].astype(float)
    residual = pred - true
    center = np.median(residual)
    return {"n": len(values),
            "median_mg_residual": float(center),
            "nmad_mg_residual": float(1.4826 * np.median(np.abs(residual - center))),
            "fraction_abs_residual_gt_1mag": float(np.mean(np.abs(residual) > 1)),
            "coverage_68_catalog_point": float(np.mean(
                (true >= records[:, 7].astype(float))
                & (true <= records[:, 9].astype(float)))),
            "coverage_95_catalog_point": float(np.mean(
                (true >= records[:, 6].astype(float))
                & (true <= records[:, 10].astype(float)))),
            "median_effective_training_n": float(np.median(records[:, 11].astype(float))),
            "mean_log_predictive_density_catalog": float(np.mean(records[:, 12].astype(float))),
            "supported_n": int(np.sum(records[:, 13].astype(bool)))}


def m3_external(training, bandwidth, mag_floor):
    reference = json.loads(M3.read_text())
    photometry = fitsio.read(str(M3_FIELD), columns=[
        "ls_id", "sigma_g", "sigma_r", "sigma_z"])
    member_ids = {star["ls_id"] for star in reference["members"]}
    errors_by_id = {int(row["ls_id"]): (
        float(row["sigma_g"]), float(row["sigma_r"]),
        float(row["sigma_z"])) for row in photometry
        if int(row["ls_id"]) in member_ids}
    x, cov = features(training)
    hull = Delaunay(x)
    means = training["spdist_mg_legacy"].astype(float)
    errors = training["spdist_sigma_mg"].astype(float)
    predictions = []
    for star in reference["members"]:
        qcolor = np.array([star["gr0"], star["rz0"]])
        qcov = color_covariance(*errors_by_id[star["ls_id"]])
        pred = predict_one(x, cov, means, errors, qcolor, qcov,
                           bandwidth, mag_floor)
        m025, m16, m50, m84, m975 = pred["quantiles"]
        predictions.append({"source_id": star["source_id"],
                            "mu_median": star["g0"] - m50,
                            "mu_68": [star["g0"] - m84, star["g0"] - m16],
                            "mu_95": [star["g0"] - m975, star["g0"] - m025],
                            "effective_training_n": pred["effective_n"],
                            "supported": bool(hull.find_simplex(qcolor) >= 0
                                              and pred["effective_n"] >= 20)})
    medians = np.array([row["mu_median"] for row in predictions])
    mu_rr = reference["reference"]["mu_mag"]
    return {"reference_mu_mag": mu_rr, "members": predictions,
            "median_photo_mu_mag": float(np.median(medians)),
            "median_photo_minus_rr_mag": float(np.median(medians) - mu_rr),
            "n_68_cover_rr": sum(row["mu_68"][0] <= mu_rr <= row["mu_68"][1]
                                 for row in predictions),
            "n_95_cover_rr": sum(row["mu_95"][0] <= mu_rr <= row["mu_95"][1]
                                 for row in predictions)}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = fitsio.read(str(SOURCE))
    mask = clean_warm_spdist(rows)
    rows = rows[mask]
    roles = np.char.strip(rows["role"].astype("U"))
    fields = np.char.strip(rows["field"].astype("U"))
    development = roles == "development"
    reserved = roles == "reserved_validation"
    if not (development.sum() > 100 and reserved.sum() > 100):
        raise ValueError("Warm giant development/reserved split is too small")
    # Tune on complete left-out development fields, never on reserved fields or M3.
    search = []
    for bandwidth in BANDWIDTHS:
        for mag_floor in MAG_FLOORS:
            field_scores = []
            for field in np.unique(fields[development]):
                query = development & (fields == field)
                train = development & (fields != field)
                if query.sum() == 0 or train.sum() < 50:
                    continue
                score = density_score(rows[train], rows[query], bandwidth,
                                      mag_floor)
                field_scores.append({"field": field, "n": int(query.sum()),
                                     "mean_log_density": score})
            weighted = sum(item["n"] * item["mean_log_density"]
                           for item in field_scores) / sum(item["n"] for item in field_scores)
            search.append({"bandwidth": bandwidth, "magnitude_floor": mag_floor,
                           "weighted_development_mean_log_density": weighted,
                           "left_out_fields": field_scores})
    best = max(search, key=lambda item: item["weighted_development_mean_log_density"])
    train = rows[development]
    evaluation = rows[reserved]
    values = estimate(evaluation, train, best["bandwidth"], best["magnitude_floor"])
    result = {"source": str(SOURCE), "model": "grz conditional label-error Gaussian kernel mixture",
              "label": "SPDist v1.0 Legacy g0 - (Gaia G0 - MG_50)",
              "selection": "RVS conservative giant, clean grz, E(B-V)<0.03, 5000<=Teff<7000 K, SPDist quality, non-cluster",
              "features": ["(g-r)0", "(r-z)0"],
              "development_n": int(development.sum()), "reserved_n": int(reserved.sum()),
              "parameter_search": search, "chosen": best,
              "reserved_summary": summarize(values),
              "reserved_supported_summary": summarize([v for v in values if v[13]]),
              "reserved_by_field": {field: summarize([v for v in values if v[1] == field])
                                    for field in np.unique(fields[reserved])},
              "m3_external": m3_external(train, best["bandwidth"],
                                         best["magnitude_floor"]),
              "compute_host": socket.gethostname()}
    write_json(OUT, result)
    print(json.dumps({"development_n": result["development_n"],
                      "reserved_n": result["reserved_n"], "chosen": best,
                      "reserved_summary": result["reserved_summary"],
                      "m3_external": result["m3_external"]}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Registered low-degree [Fe/H] feasibility test with all usable parallaxes."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.metal_poor_forward import (fit, log_probability, predictive_cdf,
                                        expected_parallax, effective_mag_variance,
                                        LOG_PAR)
from photdis.stage1 import require_compute, write_json

JOINT = [Path("data/joint_calibration_v1_evidence_v3/same_star.fits"),
         Path("data/joint_calibration_targeted_v1_evidence_v3/same_star.fits")]
OUT = Path("manifests/metal_poor_forward_v1.json")
COLUMNS = ["source_id", "ls_id", "field", "role", "trusted_blue_core",
           "parallax_layer", "gr0", "rz0", "g0", "sigma_g", "sigma_r",
           "sigma_z", "feh_cal", "desi_feh_err", "p_corr",
           "gaia_parallax_error", "desi_snr"]


def prepare():
    arrays = [fitsio.read(str(path), columns=COLUMNS) for path in JOINT]
    rows = np.concatenate(arrays)
    valid = rows["trusted_blue_core"] & (rows["source_id"] > 0)
    valid &= np.char.strip(rows["parallax_layer"].astype("U")) != "unusable"
    for name in ("gr0", "rz0", "g0", "feh_cal", "p_corr"):
        valid &= np.isfinite(rows[name])
    for name in ("sigma_g", "sigma_r", "sigma_z", "desi_feh_err",
                 "gaia_parallax_error"):
        valid &= np.isfinite(rows[name]) & (rows[name] > 0)
    rows = rows[valid]
    # Reserved identity takes priority when a star is present in two fields.
    reserved = np.char.strip(rows["role"].astype("U")) == "reserved_validation"
    order = np.lexsort((rows["ls_id"], -np.nan_to_num(rows["desi_snr"], nan=-1),
                        -reserved.astype(int), rows["source_id"]))
    _, first = np.unique(rows["source_id"][order], return_index=True)
    return rows[order[first]], {"input_usable": len(rows),
                                "identity_duplicates_removed": len(rows) - len(first)}


def metrics(data, params, with_feh):
    if not len(data):
        return {"n": 0}
    loglike = log_probability(params, data, with_feh)
    pit = predictive_cdf(params, data, with_feh)
    p = expected_parallax(params, data, with_feh)
    s2 = LOG_PAR**2 * effective_mag_variance(params, data, with_feh, 0.1)
    mean = p * np.exp(s2 / 2)
    variance = p**2 * np.exp(s2) * np.expm1(s2)
    z = (data["p_corr"] - mean) / np.sqrt(
        data["gaia_parallax_error"]**2 + variance)
    return {"n": len(data), "loglike_sum": float(loglike.sum()),
            "loglike_mean": float(loglike.mean()),
            "standardized_mean": float(np.mean(z)),
            "standardized_rms": float(np.sqrt(np.mean(z**2))),
            "pit_below_0p05": float(np.mean(pit < .05)),
            "pit_above_0p95": float(np.mean(pit > .95)),
            "negative_parallax": int(np.sum(data["p_corr"] < 0))}


def strata(data, params, with_feh):
    layers = np.char.strip(data["parallax_layer"].astype("U"))
    return {name: metrics(data[mask], params, with_feh) for name, mask in {
        "all": np.ones(len(data), bool),
        "blue_gr0_lt_0p909": data["gr0"] < .909,
        "poor_feh_lt_m1p5": data["feh_cal"] < -1.5,
        "blue_poor": (data["gr0"] < .909) & (data["feh_cal"] < -1.5),
        **{layer: layers == layer for layer in
           ("high_ge20", "mid_10_20", "weak_0_10", "negative")}}
        .items()}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    rows, dedup = prepare()
    roles = np.char.strip(rows["role"].astype("U"))
    dev = roles == "development"
    reserved = roles == "reserved_validation"
    if dev.sum() < 100 or reserved.sum() < 100:
        raise ValueError("Development/reserved role coverage too small")
    baseline, info0 = fit(rows[dev], with_feh=False)
    feh, info1 = fit(rows[dev], with_feh=True)
    sensitivity = {}
    for name, kw in {"tight_beta_prior": {"beta_prior_sigma": .5},
                     "broad_beta_prior": {"beta_prior_sigma": 10},
                     "larger_feh_error_floor": {"feh_floor": .2}}.items():
        trial, diagnostic = fit(rows[dev], with_feh=True, **kw)
        sensitivity[name] = {"parameters": trial.tolist(), "fit": diagnostic}
    field = np.char.strip(rows["field"].astype("U"))
    result = {"sources": [str(p) for p in JOINT], "selection":
              "qualified 4200-5000 K conservative dwarf, calibrated continuous [Fe/H], clean low-dust grz, usable corrected Gaia parallax including negative; Gaia identities deduplicated and reserved identities take precedence",
              "model": "Mg=a+b(gr-1)+c(rz-0.5)+d(gr-1)^2+beta([Fe/H]+1); 24-node posterior-centered Gauss-Hermite convolution of positive latent parallax with Gaussian Gaia error",
              "error_model": "g/r/z shared-band covariance, formal iron error plus 0.1 dex floor, fitted intrinsic Mg scatter; no FeH trial omits beta",
              "deduplication": dedup,
              "counts": {"all": len(rows), "development": int(dev.sum()),
                         "reserved": int(reserved.sum()),
                         "development_negative": int(np.sum(dev & (rows["p_corr"] < 0))),
                         "reserved_negative": int(np.sum(reserved & (rows["p_corr"] < 0)))},
              "models": {
                  "no_feh": {"parameters": baseline.tolist(), "fit": info0,
                             "development": strata(rows[dev], baseline, False),
                             "reserved": strata(rows[reserved], baseline, False),
                             "reserved_by_field": {f: metrics(rows[reserved & (field == f)], baseline, False)
                                                   for f in np.unique(field[reserved])}},
                  "one_feh": {"parameters": feh.tolist(), "fit": info1,
                              "development": strata(rows[dev], feh, True),
                              "reserved": strata(rows[reserved], feh, True),
                              "reserved_by_field": {f: metrics(rows[reserved & (field == f)], feh, True)
                                                    for f in np.unique(field[reserved])}}},
              "prior_error_sensitivity": sensitivity,
              "compute_host": socket.gethostname()}
    write_json(OUT, result)
    print(json.dumps({"counts": result["counts"],
                      "beta": float(feh[4]),
                      "reserved_no_feh": result["models"]["no_feh"]["reserved"]["all"],
                      "reserved_one_feh": result["models"]["one_feh"]["reserved"]["all"]}), flush=True)


if __name__ == "__main__":
    main()

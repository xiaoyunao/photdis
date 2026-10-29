#!/usr/bin/env python3
"""Inspect the training luminosity mixture at the independent M3 colors."""
from __future__ import annotations

import json
from pathlib import Path
import socket

import fitsio
import numpy as np

from giant_photo_baseline import (M3, SOURCE, clean_warm_spdist,
                                  features)
from photdis.giant_kernel import conditional_weights
from photdis.stage1 import require_compute, write_json

OUTPUT = Path("manifests/giant_m3_color_diagnostic_v1.json")
FIELD = Path("data/field_sky_ra200_210_v1/field_diagnostics.fits")
BANDWIDTH = 0.06


def weighted_quantile(values, weights, probability):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, probability)]])


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = fitsio.read(str(SOURCE))
    rows = rows[clean_warm_spdist(rows)]
    roles = np.char.strip(rows["role"].astype("U"))
    train = rows[roles == "development"]
    colors, covariance = features(train)
    members = json.loads(M3.read_text())["members"]
    errors = fitsio.read(str(FIELD), columns=["ls_id", "sigma_g", "sigma_r", "sigma_z"])
    by_id = {int(row["ls_id"]): row for row in errors}
    out = []
    for member in members:
        err = by_id[member["ls_id"]]
        qcov = np.array([[err["sigma_g"] ** 2 + err["sigma_r"] ** 2,
                          -err["sigma_r"] ** 2],
                         [-err["sigma_r"] ** 2,
                          err["sigma_r"] ** 2 + err["sigma_z"] ** 2]])
        weight, effective_n = conditional_weights(
            colors, covariance, [member["gr0"], member["rz0"]],
            qcov, BANDWIDTH)
        mg = train["spdist_mg_legacy"].astype(float)
        logg = train["desi_logg"].astype(float)
        teff = train["desi_teff"].astype(float)
        metal = train["feh_cal"].astype(float)
        available = train["abundance_calibration_valid"] & np.isfinite(metal)
        out.append({
            "source_id": member["source_id"],
            "m3_catalog_mg": float(member["g0"] - member["spdist_mu"]),
            "m3_rr_mg": float(member["g0"] - 15.041),
            "m3_logg": float(member["desi_logg"]),
            "effective_training_n": effective_n,
            "training_mg_weighted_quantiles_16_50_84": [
                weighted_quantile(mg, weight, p) for p in (.16, .5, .84)],
            "training_weight_logg_lt2p5": float(weight[logg < 2.5].sum()),
            "training_weight_logg_2p5_3p2": float(
                weight[(logg >= 2.5) & (logg < 3.2)].sum()),
            "training_weight_logg_ge3p2": float(weight[logg >= 3.2].sum()),
            "training_weight_teff_5000_5500": float(weight[teff < 5500].sum()),
            "training_weight_feh_lt_minus1p5": float(
                weight[available & (metal < -1.5)].sum()),
            "training_weight_feh_available": float(weight[available].sum()),
        })
    result = {"training_n": len(train), "bandwidth": BANDWIDTH,
              "members": out, "compute_host": socket.gethostname()}
    write_json(OUTPUT, result)
    print(json.dumps({"training_n": len(train), "members": out}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit Gaia astrometry and qualified DESI metallicity in native final fields."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.audit import rvs_metallicity
from photdis.holdouts import load_reference_catalogs
from photdis.stage1 import require_compute, write_json
from cold_v4 import PREPARED, sha
from cold_v4_policy_noise import frozen_models
from cold_v4_native_final_match import selected_rows
from cold_v4_native_transfer import LOCK, REGISTRATION

NORTH = Path("configs/cold_v4_native_transfer_north_final.json")
OUT = Path("data/cold_v4_native_transfer/final_labels.json")
EXTRA = ["source_id", "anchor", "desi_quality", "desi_label_dwarf",
         "desi_match_ambiguous", "desi_feh", "desi_feh_err", "desi_teff",
         "desi_logg", "gaia_ruwe", "gaia_parallax_error",
         "gaia_phot_g_mean_mag", "zpt_mas"]


def summary(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return {"n": len(x), "median": float(np.median(x)) if len(x) else None,
            "p05": float(np.quantile(x, .05)) if len(x) else None,
            "p95": float(np.quantile(x, .95)) if len(x) else None}


def field_result(item, references, route, forests, policy):
    name, rows = selected_rows(item["field_config"], references,
                               route, forests, policy)
    config = json.loads(Path(item["field_config"]).read_text())
    path = Path(config["output_dir"]) / "field_diagnostics.fits"
    with fitsio.FITS(str(path)) as file:
        anchor_rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        raw = file[1].read(columns=EXTRA, rows=anchor_rows)
    selected = np.isin(raw["source_id"], rows["source_id"])
    raw = raw[selected]
    if not np.array_equal(raw["source_id"], rows["source_id"]):
        raise ValueError("Selected source order differs from field diagnostics")
    p = rows["p_corr"]
    err = np.asarray(raw["gaia_parallax_error"], float)
    ruwe = np.asarray(raw["gaia_ruwe"], float)
    zpt = np.asarray(raw["zpt_mas"], float)
    residual = rows["residual"]
    feh, feh_domain = rvs_metallicity(raw["desi_feh"], raw["desi_teff"],
                                       raw["desi_logg"])
    desi = (raw["desi_quality"] & raw["desi_label_dwarf"] &
            ~raw["desi_match_ambiguous"] & feh_domain &
            np.isfinite(raw["desi_feh_err"]) & (raw["desi_feh_err"] > 0))
    strata = {
        "ruwe_le_1p1": ruwe <= 1.1,
        "ruwe_gt_1p1": ruwe > 1.1,
        "parallax_ge_2_ruwe_le_1p1": (p >= 2) & (ruwe <= 1.1),
        "parallax_ge_2_ruwe_gt_1p1": (p >= 2) & (ruwe > 1.1),
        "desi_dwarf_calibrated": desi,
        "desi_feh_lt_m0p5": desi & (feh < -.5),
        "desi_feh_m0p5_to_0": desi & (feh >= -.5) & (feh < 0),
        "desi_feh_ge_0": desi & (feh >= 0),
    }
    return {"field": name, "selected_count": len(residual),
            "gaia": {"ruwe": summary(ruwe), "parallax_error_mas": summary(err),
                      "corrected_parallax_snr": summary(p / err),
                      "applied_zpt_mas": summary(zpt),
                      "phot_g_mean_mag": summary(raw["gaia_phot_g_mean_mag"])},
            "qualified_desi_dwarf_feh": summary(feh[desi]),
            "strata": {key: {"residual": summary(residual[mask]),
                             "feh": summary(feh[mask]) if key.startswith("desi") else None}
                       for key, mask in strata.items()}}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    original = json.loads(REGISTRATION.read_text())
    north = json.loads(NORTH.read_text())
    if (north["parent_registration_sha256"] != sha(REGISTRATION)
            or original["prepared_sha256"] != sha(PREPARED)
            or original["policy_lock_sha256"] != sha(LOCK)):
        raise ValueError("Registration or frozen model/policy lock changed")
    route, forests = frozen_models()
    policy = json.loads(LOCK.read_text())["chosen_policy"]
    references = load_reference_catalogs("data/reference")
    configs = original["independent_final_fields"] + north["independent_final_fields"]
    result = {"method": "frozen selected native cold anchors; DESI quality and "
                       "valid metallicity-calibration domain required",
              "compute_host": socket.gethostname(),
              "south_registration_sha256": sha(REGISTRATION),
              "north_registration_sha256": sha(NORTH),
              "policy_lock_sha256": sha(LOCK),
              "fields": [field_result(item, references, route, forests, policy)
                         for item in configs]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Describe residuals versus Gaia tangential motion in native final fields."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
from astropy.coordinates import SkyCoord
import astropy.units as u
import fitsio
import numpy as np

from photdis.holdouts import load_reference_catalogs
from photdis.stage1 import require_compute, write_json
from cold_v4 import PREPARED, sha
from cold_v4_policy_noise import frozen_models
from cold_v4_native_final_match import selected_rows
from cold_v4_native_transfer import LOCK, REGISTRATION, spatial_bootstrap

NORTH = Path("configs/cold_v4_native_transfer_north_final.json")
OUT = Path("data/cold_v4_native_transfer/final_kinematics.json")


def group_result(rows, mask, seed):
    residual = rows["residual"][mask]
    if not len(residual):
        return {"n": 0}
    center = float(np.median(residual))
    return {"n": len(residual), "median_residual_mag": center,
            "nmad_residual_mag": float(1.4826 * np.median(abs(residual - center))),
            "sky_block": spatial_bootstrap(residual, rows["ra"][mask],
                       rows["dec"][mask], seed)}


def field_result(item, references, route, forests, policy, seed):
    name, rows = selected_rows(item["field_config"], references,
                               route, forests, policy)
    config = json.loads(Path(item["field_config"]).read_text())
    path = Path(config["output_dir"]) / "field_diagnostics.fits"
    with fitsio.FITS(str(path)) as file:
        anchor_rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        extras = file[1].read(columns=["source_id", "gaia_pmra", "gaia_pmdec"],
                              rows=anchor_rows)
    extras = extras[np.isin(extras["source_id"], rows["source_id"])]
    if not np.array_equal(extras["source_id"], rows["source_id"]):
        raise ValueError("Selected source order differs from Gaia motion rows")
    pm = np.hypot(np.asarray(extras["gaia_pmra"], float),
                  np.asarray(extras["gaia_pmdec"], float))
    vt = 4.74047 * pm / rows["p_corr"]
    b = SkyCoord(ra=rows["ra"] * u.deg,
                 dec=rows["dec"] * u.deg).galactic.b.deg
    z_abs_pc = 1000 * abs(np.sin(np.deg2rad(b))) / rows["p_corr"]
    finite = np.isfinite(vt) & np.isfinite(z_abs_pc)
    groups = {
        "vt_lt_40": finite & (vt < 40),
        "vt_40_to_80": finite & (vt >= 40) & (vt < 80),
        "vt_ge_80": finite & (vt >= 80),
        "p_ge_2_vt_lt_40": finite & (rows["p_corr"] >= 2) & (vt < 40),
        "p_ge_2_vt_40_to_80": finite & (rows["p_corr"] >= 2) &
                               (vt >= 40) & (vt < 80),
        "p_ge_2_vt_ge_80": finite & (rows["p_corr"] >= 2) & (vt >= 80),
        "z_abs_lt_200_pc": finite & (z_abs_pc < 200),
        "z_abs_200_to_400_pc": finite & (z_abs_pc >= 200) & (z_abs_pc < 400),
        "z_abs_ge_400_pc": finite & (z_abs_pc >= 400),
    }
    return {"field": name, "selected_count": len(vt),
            "finite_motion_count": int(finite.sum()),
            "median_tangential_speed_kms": float(np.median(vt[finite])),
            "median_abs_height_pc": float(np.median(z_abs_pc[finite])),
            "groups": {key: group_result(rows, mask, seed + index)
                       for index, (key, mask) in enumerate(groups.items())}}


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
    references = load_reference_catalogs("data/reference")
    policy = json.loads(LOCK.read_text())["chosen_policy"]
    configs = original["independent_final_fields"] + north["independent_final_fields"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, {"method": "Gaia proper-motion tangential speed and absolute "
                          "height diagnostics on frozen selected cold anchors; "
                          "no population membership inferred",
                     "compute_host": socket.gethostname(),
                     "south_registration_sha256": sha(REGISTRATION),
                     "north_registration_sha256": sha(NORTH),
                     "policy_lock_sha256": sha(LOCK),
                     "fields": [field_result(item, references, route, forests,
                                             policy, 20261210 + index * 20)
                                for index, item in enumerate(configs)]})


if __name__ == "__main__":
    main()

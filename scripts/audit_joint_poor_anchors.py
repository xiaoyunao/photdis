#!/usr/bin/env python3
"""List the scarce metal-poor geometric anchors in a joint calibration sample."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.stage1 import require_compute, write_json

COLUMNS = ("field", "role", "source_id", "ls_id", "class_evidence",
           "parallax_layer", "clean_low_dust", "feh_cal_valid", "feh_cal",
           "desi_feh_err", "desi_teff", "desi_teff_err", "desi_logg",
           "desi_logg_err", "gr0", "rz0", "r", "bp_rp_checked",
           "p_corr", "gaia_parallax_error", "mg_high_snr",
           "sigma_mg_high_snr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = fitsio.read(str(args.source), columns=COLUMNS)
    labels = np.char.strip(rows["class_evidence"].astype("U"))
    layers = np.char.strip(rows["parallax_layer"].astype("U"))
    poor = ((labels == "cool_main_sequence") & (layers == "high_ge20")
            & rows["clean_low_dust"] & rows["feh_cal_valid"]
            & (rows["feh_cal"] < -1.5))
    selected = rows[poor]
    order = np.argsort(selected["gr0"])
    selected = selected[order]
    result = []
    for row in selected:
        result.append({
            "field": str(row["field"]).strip(),
            "role": str(row["role"]).strip(),
            "gaia_source_id": int(row["source_id"]),
            "legacy_id": int(row["ls_id"]),
            "gr0": float(row["gr0"]), "rz0": float(row["rz0"]),
            "r": float(row["r"]), "bp_rp": float(row["bp_rp_checked"]),
            "teff": float(row["desi_teff"]),
            "teff_error": float(row["desi_teff_err"]),
            "logg": float(row["desi_logg"]),
            "logg_error": float(row["desi_logg_err"]),
            "feh_cal": float(row["feh_cal"]),
            "feh_formal_error": float(row["desi_feh_err"]),
            "parallax_corrected_mas": float(row["p_corr"]),
            "parallax_error_mas": float(row["gaia_parallax_error"]),
            "mg": float(row["mg_high_snr"]),
            "mg_formal_error": float(row["sigma_mg_high_snr"]),
        })
    manifest = {
        "selection": "qualified 4200-5000 K conservative main sequence, clean low-dust grz, calibrated FeH<-1.5, corrected Gaia S/N>=20",
        "input_rows": len(rows),
        "selected_rows": len(result),
        "blue_below_gr0_0p909": sum(item["gr0"] < .909 for item in result),
        "by_role": {role: sum(item["role"] == role for item in result)
                    for role in ("development", "reserved_validation")},
        "objects": result,
        "compute_host": socket.gethostname(),
    }
    write_json(args.output, manifest)
    print({key: manifest[key] for key in ("input_rows", "selected_rows",
                                         "blue_below_gr0_0p909", "by_role")})


if __name__ == "__main__":
    main()

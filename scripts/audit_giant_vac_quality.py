#!/usr/bin/env python3
"""Report each exact-match giant distance quality loss separately."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.stage1 import require_compute, write_json

SOURCE = Path("data/giant_vac_v3/same_star.fits")
OUT = Path("manifests/giant_vac_quality_audit_v3.json")


def count(mask):
    return int(np.sum(mask))


def distribution(values):
    values = np.asarray(values, float)
    good = values[np.isfinite(values)]
    return {"n": len(values), "finite": len(good),
            "min": float(np.min(good)) if len(good) else None,
            "median": float(np.median(good)) if len(good) else None,
            "p90": float(np.percentile(good, 90)) if len(good) else None,
            "max": float(np.max(good)) if len(good) else None}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = fitsio.read(str(SOURCE), columns=[
        "clean_low_dust", "specdis_match", "specdis_valid",
        "specdis_binary_flag", "specdis_dist_kpc", "specdis_disterr_kpc",
        "spdist_match", "spdist_valid", "spdist_flag_good",
        "spdist_qflag_input", "spdist_mg50_gaia", "spdist_g0_gaia",
        "g0", "sigma_g", "desi_teff", "desi_logg", "desi_snr"])
    clean = rows["clean_low_dust"]
    m = rows["specdis_match"]
    d = rows["specdis_dist_kpc"]
    e = rows["specdis_disterr_kpc"]
    single = rows["specdis_binary_flag"] == 1
    dist = np.isfinite(d) & (d > 0)
    err = np.isfinite(e) & (e > 0)
    relative = err & dist & (e < d)
    photo = np.isfinite(rows["g0"]) & np.isfinite(rows["sigma_g"])
    spec = {"matched": count(m), "matched_clean": count(m & clean),
            "binary_flag": {str(int(flag)): count(m & (rows["specdis_binary_flag"] == flag))
                            for flag in np.unique(rows["specdis_binary_flag"][m])},
            "distance": distribution(d[m]), "distance_error": distribution(e[m]),
            "fractional_error": distribution(e[m & dist] / d[m & dist]),
            "waterfall_clean": {"matched": count(m & clean),
                                "single": count(m & clean & single),
                                "finite_positive_distance": count(m & clean & single & dist),
                                "finite_positive_error": count(m & clean & single & dist & err),
                                "error_less_than_distance": count(m & clean & single & relative),
                                "usable_legacy_g": count(m & clean & single & relative & photo),
                                "versioned_valid": count(clean & rows["specdis_valid"])}}
    m = rows["spdist_match"]
    good = rows["spdist_flag_good"]
    within = rows["spdist_qflag_input"]
    sp = {"matched": count(m), "matched_clean": count(m & clean),
          "flag_good": count(m & good), "qflag_input": count(m & within),
          "both_flags": count(m & good & within),
          "both_flags_clean": count(m & clean & good & within),
          "g0": distribution(rows["spdist_g0_gaia"][m]),
          "mg50": distribution(rows["spdist_mg50_gaia"][m]),
          "versioned_valid": count(clean & rows["spdist_valid"])}
    out = {"source": str(SOURCE), "n_giants": len(rows),
           "specdis": spec, "spdist": sp, "compute_host": socket.gethostname()}
    write_json(OUT, out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

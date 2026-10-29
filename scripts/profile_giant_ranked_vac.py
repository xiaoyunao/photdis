#!/usr/bin/env python3
"""Summarize quality-valid cool giant labels by frozen field role."""
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

SOURCE = Path("data/giant_ranked_vac_v1/same_star.fits")
OUTPUT = Path("manifests/giant_ranked_vac_profile_v2.json")


def stats(rows):
    clean = rows["clean_low_dust"] & ~rows["cmd_spectral_conflict"]
    teff = rows["desi_teff"]
    k_all = (teff >= 4200) & (teff < 5000)
    k = clean & k_all
    spec = k & rows["specdis_valid"]
    sp = k & rows["spdist_valid"]
    high_spec = spec & (rows["desi_snr"] >= 60)
    result = {
        "all_giants": int(len(rows)),
        "clean_low_dust_giants": int(clean.sum()),
        "k_4200_5000_all": int(k_all.sum()),
        "k_grz_clean": int((k_all & rows["grz_clean"]).sum()),
        "k_ebv_lt0p03": int((k_all & np.isfinite(rows["ebv"])
                              & (rows["ebv"] < .03)).sum()),
        "k_cmd_spectral_conflict": int((k_all & rows["cmd_spectral_conflict"]).sum()),
        "k_4200_5000_clean": int(k.sum()),
        "k_specdis_valid": int(spec.sum()),
        "k_specdis_snr_ge60": int(high_spec.sum()),
        "k_spdist_valid": int(sp.sum()),
        "k_both_valid": int((spec & sp).sum()),
        "k_spdist_mg_sigma_median": (
            float(np.median(rows["spdist_sigma_mg"][sp])) if sp.any() else None),
        "k_specdis_fractional_distance_error_median": (
            float(np.median(rows["specdis_disterr_kpc"][spec]
                            / rows["specdis_dist_kpc"][spec])) if spec.any() else None),
        "k_spdist_mg_range": ([float(np.min(rows["spdist_mg_legacy"][sp])),
                               float(np.max(rows["spdist_mg_legacy"][sp]))]
                              if sp.any() else None),
    }
    return result


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = fitsio.read(str(SOURCE))
    roles = np.char.strip(rows["role"].astype("U"))
    fields = np.char.strip(rows["field"].astype("U"))
    result = {
        "source": str(SOURCE), "all": stats(rows),
        "by_role": {role: stats(rows[roles == role])
                    for role in ("development", "reserved_validation")},
        "by_field": {field: stats(rows[fields == field])
                     for field in np.unique(fields)},
        "compute_host": socket.gethostname(),
    }
    write_json(OUTPUT, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize label uncertainty, catalog disagreement and cluster overlap."""
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
OUT = Path("manifests/giant_vac_profile_v3.json")


def median_nmad(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    center = np.median(x)
    return {"n": len(x), "median": float(center),
            "nmad": float(1.4826 * np.median(np.abs(x - center))),
            "p16": float(np.percentile(x, 16)),
            "p84": float(np.percentile(x, 84))}


def profile(rows, mask):
    spec = mask & rows["specdis_valid"]
    sp = mask & rows["spdist_valid"]
    both = spec & sp
    disagreement = (rows["specdis_mg_legacy"][both]
                    - rows["spdist_mg_legacy"][both])
    combined = np.hypot(rows["specdis_sigma_mg"][both],
                        rows["spdist_sigma_mg"][both])
    snr = rows["desi_snr"]
    snr_bins = {"lt20": snr < 20, "20_60": (snr >= 20) & (snr < 60),
                "60_100": (snr >= 60) & (snr < 100), "ge100": snr >= 100}
    return {"n": int(mask.sum()),
            "specdis_match": int((mask & rows["specdis_match"]).sum()),
            "spdist_match": int((mask & rows["spdist_match"]).sum()),
            "specdis": int(spec.sum()),
            "spdist": int(sp.sum()), "both": int(both.sum()),
            "specdis_by_snr": {name: {"n": int((spec & in_bin).sum()),
                "fractional_distance_error": median_nmad(
                    rows["specdis_disterr_kpc"][spec & in_bin] /
                    rows["specdis_dist_kpc"][spec & in_bin])}
                for name, in_bin in snr_bins.items()},
            "specdis_fractional_distance_error": median_nmad(
                rows["specdis_disterr_kpc"][spec] / rows["specdis_dist_kpc"][spec]),
            "specdis_sigma_legacy_mg": median_nmad(rows["specdis_sigma_mg"][spec]),
            "spdist_sigma_legacy_mg": median_nmad(rows["spdist_sigma_mg"][sp]),
            "specdis_minus_spdist_mg": median_nmad(disagreement),
            "abs_disagreement_gt_2sigma": float(np.mean(np.abs(disagreement) > 2 * combined))
            if len(disagreement) else None}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = fitsio.read(str(SOURCE))
    clean = rows["clean_low_dust"] & ~rows["cmd_spectral_conflict"]
    teff = rows["desi_teff"]
    groups = {"all_clean": clean,
              "cool_lt4200": clean & (teff < 4200),
              "k_4200_5000": clean & (teff >= 4200) & (teff < 5000),
              "warm_5000_7000": clean & (teff >= 5000) & (teff < 7000),
              "logg_lt1p8": clean & (rows["desi_logg"] < 1.8),
              "abundance_outside_calibration": clean & ~rows["abundance_calibration_valid"]}
    names = json.loads(Path("data/reference/holdout_group_names.json").read_text())
    cluster = np.genfromtxt("data/reference/vasiliev_baumgardt_2021_clusters.txt",
                            names=True, dtype=None, encoding="utf-8")
    system = []
    for gid in np.unique(rows["cluster_group"][clean & (rows["cluster_group"] > 0)]):
        mask = clean & (rows["cluster_group"] == gid)
        if gid >= 100000:
            ref = cluster[int(gid) - 100000]
            distance, error = float(ref["dist"]), float(ref["disterr"])
        else:
            distance = error = None
        system.append({"group": int(gid), "name": names[str(int(gid))]["name"],
                       "reference_distance_kpc": distance,
                       "reference_error_kpc": error,
                       "profile": profile(rows, mask)})
    result = {"source": str(SOURCE),
              "groups": {name: profile(rows, mask) for name, mask in groups.items()},
              "external_member_systems": system,
              "compute_host": socket.gethostname()}
    write_json(OUT, result)
    print(json.dumps({"groups": result["groups"],
                      "external_member_systems": system}), flush=True)


if __name__ == "__main__":
    main()

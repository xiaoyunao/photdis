#!/usr/bin/env python3
"""Audit the newly registered NGC 7006 DECam development stage."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps"), str(ROOT / "scripts")]
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cluster_rgb_inventory import SYSTEMS
from cluster_rgb_system_audit import (RG, RR, RZ,
                                      describe, local_residuals, sequence)
from photdis.stage1 import require_compute, write_json


def subset(tab, stage, band):
    phase = tab["phase"].astype("U")
    quality = "gr_clean" if band == "gr" else "grz_clean_same_row"
    return ((tab["pmember"] > .9) & (phase == stage) & tab[quality]
            & np.isfinite(tab["g0"]+tab["gr0"])
            & (np.isfinite(tab["rz0"]) if band == "grz" else True))


def view(name, phase, band):
    tab = fitsio.read(f"data/cluster_rgb_gr_v2/{name}.fits")
    use = subset(tab, phase, band)
    selected = tab[use]
    mg = selected["g0"].astype(float)-SYSTEMS[name]["mu"]
    gr = selected["gr0"].astype(float)
    rz = selected["rz0"].astype(float) if band == "grz" else None
    return selected, mg, gr, rz


def summarize_stage(name, phase, band):
    tab, mg, gr, rz = view(name, phase, band)
    sg, sr = tab["sigma_g"].astype(float), tab["sigma_r"].astype(float)
    gr_points = sequence(mg, gr, minimum=3 if name == "ngc7006" else 8)
    gr_resid, gr_expected = local_residuals(mg, gr, sg, sg, sr, gr_points, True)
    result = {"n": len(tab), "mg": describe(mg), "gr": describe(gr),
              "mg_minmax": [float(np.min(mg)), float(np.max(mg))] if len(mg) else [],
              "gr_points": gr_points,
              "gr_local_residual": describe(gr_resid),
              "gr_measurement_sigma": describe(gr_expected),
              "radius": describe(tab["radius_arcmin"]),
              "quality": {k: describe(tab[k]) for k in (
                  "fracmasked_g", "fracmasked_r", "rchisq_g", "rchisq_r")}}
    for key, values in (("radius", tab["radius_arcmin"]),
                        ("fracmasked_gr", tab["fracmasked_g"]+tab["fracmasked_r"]),
                        ("rchisq_gr", tab["rchisq_g"]+tab["rchisq_r"])):
        if len(values):
            pivot = float(np.median(values))
            result[f"gr_residual_by_{key}"] = {
                "split": pivot,
                "low": describe(gr_resid[values <= pivot]),
                "high": describe(gr_resid[values > pivot])}
    if rz is not None:
        sz = tab["sigma_z"].astype(float)
        rz_points = sequence(mg, rz, minimum=3 if name == "ngc7006" else 8)
        rz_resid, rz_expected = local_residuals(mg, rz, sg, sr, sz, rz_points, False)
        result.update({"rz": describe(rz), "rz_points": rz_points,
                       "rz_local_residual": describe(rz_resid),
                       "rz_measurement_sigma": describe(rz_expected),
                       "measurement_covariance": {
                           "cov_gr_rz_median": float(np.median(-sr**2)),
                           "cov_mg_gr_median": float(np.median(sg**2))}})
    return result


def overlap(a, b):
    lo = max(a["mg"]["p05"], b["mg"]["p05"])
    hi = min(a["mg"]["p95"], b["mg"]["p95"])
    return {"mg_5_95_overlap": [lo, hi], "exists": bool(hi > lo)}


def plot(path, data):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for name, color in (("m53", "tab:blue"), ("ngc7006", "tab:orange")):
        for j, (band, key) in enumerate((("gr", "gr_points"), ("grz", "rz_points"))):
            pts = data[name][band][key]
            if pts:
                ax[j].plot([p["color"] for p in pts], [p["mg"] for p in pts],
                           "o-", ms=3, color=color,
                           label=f"{name.upper()} upper RGB ({data[name][band]['n']})")
    for axis, label in zip(ax, ("(g−r)₀", "(r−z)₀")):
        axis.set(xlabel=label, ylabel="M_g from independent cluster modulus")
        axis.invert_yaxis()
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    require_compute()
    out = Path("manifests/cluster_rgb_ngc7006_development_v1.json")
    if out.exists():
        raise FileExistsError(out)
    data = {name: {band: summarize_stage(name, "upper_rgb", band)
                   for band in ("gr", "grz")}
            for name in ("m53", "ngc7006")}
    result = {"role": "ngc7006_new_experiment_development_only",
              "historical_ngc7006_test": "frozen_old_pooled_A_B_result",
              "samples": data,
              "upper_rgb_overlap": {band: overlap(data["m53"][band],data["ngc7006"][band])
                                    for band in ("gr", "grz")},
              "ngc7006_shared_scale": {
                  "mu_sigma": float(np.hypot(SYSTEMS["ngc7006"]["mu_stat"],
                                              SYSTEMS["ngc7006"]["mu_sys"])),
                  "dust_trial_sigma_ebv": 0.05,
                  "dust_color_shift_gr": (RG-RR)*0.05,
                  "dust_color_shift_rz": (RR-RZ)*0.05,
                  "dust_mg_shift": RG*0.05},
              "figure": "figures/cluster_rgb/05_ngc7006_upper_rgb_development.png",
              "compute_host": socket.gethostname()}
    plot(Path(result["figure"]), data)
    write_json(out, result)
    print(json.dumps({"n": {n: {b: data[n][b]["n"] for b in data[n]}
                            for n in data}, "overlap": result["upper_rgb_overlap"]}),flush=True)


if __name__ == "__main__":
    main()

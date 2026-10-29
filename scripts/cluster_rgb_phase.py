#!/usr/bin/env python3
"""Audit broad Gaia-CMD phase candidates and measured Legacy RGB loci."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photdis.stage1 import require_compute, write_json

CONFIG = Path("configs/cluster_rgb_phase_regions_v1.json")
PHASES = ("upper_rgb", "lower_rgb", "hb", "possible_agb", "main_sequence",
          "ambiguous", "invalid_gaia_cmd")


def classify(gaia_g_minus_mu, bp_rp, cfg):
    """Return mutually exclusive Gaia-CMD candidates, independent of grz."""
    mg = np.asarray(gaia_g_minus_mu, float)
    color = np.asarray(bp_rp, float)
    out = np.full(len(mg), "ambiguous", dtype="U20")
    finite = np.isfinite(mg) & np.isfinite(color) & (color > -1) & (color < 5)
    out[~finite] = "invalid_gaia_cmd"
    boundary = cfg["rgb_color_lower_at_zero"] + cfg["rgb_color_slope_per_mag"]*mg
    upper = finite & (mg <= cfg["rgb_bright_max_gaia_g_minus_mu"]) & (color >= boundary)
    lower = (finite & (mg > cfg["rgb_bright_max_gaia_g_minus_mu"])
             & (mg <= cfg["rgb_lower_max_gaia_g_minus_mu"]) & (color >= boundary))
    out[upper] = "upper_rgb"
    out[lower] = "lower_rgb"
    hb = (finite & (mg >= cfg["hb_gaia_g_minus_mu"][0])
          & (mg <= cfg["hb_gaia_g_minus_mu"][1])
          & (color < cfg["hb_bp_rp_max"]) & ~lower)
    out[hb] = "hb"
    possible_agb = (finite & (mg <= cfg["rgb_bright_max_gaia_g_minus_mu"])
                    & (color >= cfg["possible_agb_bp_rp_min"])
                    & (color < boundary) & ~upper)
    out[possible_agb] = "possible_agb"
    ms = finite & (mg >= cfg["main_sequence_gaia_g_minus_mu_min"])
    out[ms] = "main_sequence"
    return out


def summary(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    center = np.median(x)
    return {"n": int(len(x)), "median": float(center),
            "nmad": float(1.4826*np.median(np.abs(x-center))),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}


def locus(mg, color, *, step=.5, minimum=5):
    bins = np.arange(-3.5, 4.01, step)
    points = []
    for left, right in zip(bins[:-1], bins[1:]):
        keep = ((mg >= left) & (mg < right) & np.isfinite(mg)
                & np.isfinite(color))
        if keep.sum() < minimum:
            continue
        points.append({"mg_bin": [float(left), float(right)],
                       "n": int(keep.sum()),
                       "mg_median": float(np.median(mg[keep])),
                       "color_median": float(np.median(color[keep])),
                       "color_p16": float(np.percentile(color[keep], 16)),
                       "color_p84": float(np.percentile(color[keep], 84))})
    return points


def phase_counts(tab, mask):
    core = mask & (tab["pmember"] > .9)
    lower = mask & (tab["pmember"] >= .5) & (tab["pmember"] <= .9)
    optical = core & tab["grz_base"]
    clean = core & tab["grz_clean"]
    return {"p_gt_0p9": int(core.sum()),
            "p_0p5_to_0p9": int(lower.sum()),
            "grz_base": int(optical.sum()),
            "fraction_clean": int((optical & tab["grz_fraction_clean"]).sum()),
            "mask_clean": int((optical & tab["grz_mask_clean"]).sum()),
            "grz_clean": int(clean.sum()),
            "both_fraction_mask_fail": int((optical & ~tab["grz_fraction_clean"]
                                            & ~tab["grz_mask_clean"]).sum()),
            "clean_w1": int((clean & tab["w1_clean"]).sum()),
            "clean_w1w2": int((clean & tab["w1_clean"] & tab["w2_clean"]).sum()),
            "gaia_g_rejected": summary(tab["gaia_g"][core & ~tab["grz_clean"]]),
            "gaia_g_retained": summary(tab["gaia_g"][clean]),
            "radius_arcmin_rejected": summary(tab["radius_arcmin"][core & ~tab["grz_clean"]]),
            "radius_arcmin_retained": summary(tab["radius_arcmin"][clean]),
            "ebv_clean": summary(tab["ebv"][clean])}


def radial_quality(tab, mask):
    core = mask & (tab["pmember"] > .9)
    bins = [0, 1, 2, 5, 10, 20, 60, 180]
    out = []
    for left, right in zip(bins[:-1], bins[1:]):
        take = core & (tab["radius_arcmin"] >= left) & (tab["radius_arcmin"] < right)
        if take.sum():
            out.append({"radius_arcmin": [left, right], "all": int(take.sum()),
                        "base": int((take & tab["grz_base"]).sum()),
                        "clean": int((take & tab["grz_clean"]).sum())})
    return out


def plot_phase(path, tab, phase, mu, system, loci):
    high = tab["pmember"] > .9
    clean = high & tab["grz_clean"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    colors = {"upper_rgb":"tab:red", "lower_rgb":"tab:orange", "hb":"tab:blue",
              "possible_agb":"tab:purple", "main_sequence":"0.75", "ambiguous":"0.65"}
    for name in ("main_sequence", "ambiguous", "hb", "possible_agb", "lower_rgb", "upper_rgb"):
        take = high & (phase == name)
        ax[0].scatter(tab["bp_rp"][take], tab["gaia_g"][take]-mu,
                      s=3 if name not in ("upper_rgb", "lower_rgb") else 6,
                      alpha=.5, color=colors[name], label=f"{name}: {take.sum()}")
    ax[0].legend(fontsize=6, markerscale=2)
    ax[0].set(xlabel="Gaia BP−RP", ylabel="Gaia G − μ", title="CMD phase candidates")
    for j, col in enumerate(("gr0", "rz0"), start=1):
        for name in ("lower_rgb", "upper_rgb"):
            take = clean & (phase == name)
            ax[j].scatter(tab[col][take], tab["g0"][take]-mu,
                          s=7, alpha=.4, color=colors[name], label=name)
        points = loci[col]
        if points:
            ax[j].plot([v["color_median"] for v in points],
                       [v["mg_median"] for v in points],
                       "ko-", ms=3, lw=1, label="RGB bin medians")
        ax[j].set(xlabel="(g−r)₀" if col == "gr0" else "(r−z)₀",
                  ylabel="Legacy g₀ − μ", title="Actual clean Legacy RGB candidates")
        ax[j].legend(fontsize=7)
    for a in ax:
        a.set_ylim(6, -4)
        a.grid(alpha=.15)
    fig.suptitle(f"{system}: broad Gaia phase regions, Pmember>0.9")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    require_compute()
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=json.loads(CONFIG.read_text())["systems"])
    args = parser.parse_args()
    cfg_all = json.loads(CONFIG.read_text())
    cfg = cfg_all["systems"][args.system]
    inp = Path("data")/f"cluster_rgb_{args.system}_v1"/"members.fits"
    manifest = Path("manifests")/f"cluster_rgb_{args.system}_phase_v1.json"
    figure = Path("figures/cluster_rgb")/f"02_phase_locus_{args.system}.png"
    if manifest.exists():
        raise FileExistsError(manifest)
    tab = fitsio.read(str(inp))
    inventory = json.loads((Path("manifests")/f"cluster_rgb_{args.system}_inventory_v1.json").read_text())
    mu = inventory["external_modulus"]["mu_mag"]
    phase = classify(tab["gaia_g"]-mu, tab["bp_rp"], cfg)
    counts = {name: phase_counts(tab, phase == name) for name in PHASES}
    if inventory["role"] == "reserved_whole_system":
        out = {"system": args.system, "role": inventory["role"],
               "membership_probability_core": ">0.9",
               "phase_config": str(CONFIG), "phase_definition": cfg,
               "phase_caveat": cfg_all["stage_caveat"],
               "external_modulus": inventory["external_modulus"],
               "phase_counts": counts,
               "rgb_candidate_radius_quality": radial_quality(
                   tab, np.isin(phase, ("upper_rgb", "lower_rgb"))),
               "data_file": str(inp), "figure": inventory["figure"],
               "holdout_state": "Gaia CMD and DR11 quality counts only; Legacy luminosities sealed",
               "compute_host": socket.gethostname()}
        write_json(manifest, out)
        print(json.dumps({"phase_counts": {k:v["grz_clean"] for k,v in counts.items()}}),
              flush=True)
        return
    rgb = np.isin(phase, ("upper_rgb", "lower_rgb")) & (tab["pmember"] > .9)
    core = rgb & tab["grz_clean"] & np.isfinite(tab["g0"])
    mg = tab["g0"][core] - mu
    loci = {color: locus(mg, tab[color][core]) for color in ("gr0", "rz0")}
    ag = tab["g"][core]-tab["g0"][core]
    ar = tab["r"][core]-tab["r0"][core]
    az = tab["z"][core]-tab["z0"][core]
    ebv = tab["ebv"][core]
    normal = np.isfinite(ebv) & (ebv > .001)
    extinction = {"ebv": summary(ebv),
                  "a_g_over_ebv": summary(ag[normal]/ebv[normal]),
                  "a_r_over_ebv": summary(ar[normal]/ebv[normal]),
                  "a_z_over_ebv": summary(az[normal]/ebv[normal])}
    spectral = None
    source_file = {"m3": "giant_m3_reference_v1.json",
                   "ngc2419": "giant_ngc2419_reference_v1.json"}.get(args.system)
    if source_file:
        source_spectrum = json.loads((Path("manifests")/source_file).read_text())
        spectral_rows = source_spectrum.get("members", source_spectrum.get("stars", []))
        sid_phase = {int(sid): str(p) for sid,p in zip(tab["source_id"],phase)}
        spectral = {"qualified_giant_phases": {name: sum(
            sid_phase.get(int(s["source_id"])) == name for s in spectral_rows)
            for name in PHASES}}
    out = {"system": args.system, "role": inventory["role"],
           "membership_probability_core": ">0.9",
           "phase_config": str(CONFIG), "phase_definition": cfg,
           "phase_caveat": cfg_all["stage_caveat"],
           "external_modulus": inventory["external_modulus"],
           "phase_counts": counts,
           "rgb_clean_legacy": {"n": int(core.sum()), "mg": summary(mg),
                                "gr0": summary(tab["gr0"][core]),
                                "rz0": summary(tab["rz0"][core]),
                                "upper": int((core & (phase == "upper_rgb")).sum()),
                                "lower": int((core & (phase == "lower_rgb")).sum()),
                                "radius_quality": radial_quality(tab, rgb)},
           "locus_by_legacy_mg": loci, "extinction": extinction,
           "spectral_stage_check": spectral,
           "data_file": str(inp), "figure": str(figure),
           "compute_host": socket.gethostname()}
    plot_phase(figure, tab, phase, mu, args.system, loci)
    write_json(manifest, out)
    print(json.dumps({"phase_counts": {k:v["grz_clean"] for k,v in counts.items()},
                      "rgb_mg": out["rgb_clean_legacy"]["mg"]}),flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Band-matched RGB quality and native-system sequence audit on Slurm compute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps"), str(ROOT / "scripts")]
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cluster_rgb_inventory import SYSTEMS, matched_sweeps
from cluster_rgb_phase import CONFIG, classify
from photdis.stage1 import legacy_ids, require_compute, structured, write_json

CORE_PHASES = {"m3": ("lower_rgb",), "m53": ("upper_rgb", "lower_rgb"),
               "ngc2419": ("upper_rgb",)}
SYSTEM_LABEL = {"m3": "DECam", "m53": "DECam", "ngc2419": "BASS/MzLS",
                "ngc7006": "DECam"}
BIN_EDGES = np.arange(-3.5, 4.01, .4)
RG, RR, RZ = 3.214, 2.165, 1.211
DUST_SCALE_TRIAL = {"m3": .005, "m53": .010, "ngc2419": .020}


def describe(x):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "nmad": float(1.4826*np.median(np.abs(a-np.median(a))))}


def sequence(mg, color, minimum=8):
    pts = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        use = np.isfinite(mg) & np.isfinite(color) & (mg >= lo) & (mg < hi)
        if use.sum() >= minimum:
            pts.append({"mg": float(np.median(mg[use])),
                        "color": float(np.median(color[use])),
                        "n": int(use.sum()),
                        "p16": float(np.percentile(color[use], 16)),
                        "p84": float(np.percentile(color[use], 84))})
    return pts


def local_residuals(mg, color, sigma_g, sigma_first, sigma_second,
                    points, cov_g_color):
    """Residual and propagated error for color minus local color(Mg) curve."""
    if len(points) < 2:
        return np.full(len(mg), np.nan), np.full(len(mg), np.nan)
    xp = np.array([p["mg"] for p in points])
    yp = np.array([p["color"] for p in points])
    slope = np.interp(mg, xp, np.gradient(yp, xp))
    pred = np.interp(mg, xp, yp)
    inside = np.isfinite(mg) & (mg >= xp[0]) & (mg <= xp[-1])
    resid = np.where(inside, color-pred, np.nan)
    # Mg = g0 - common cluster modulus. g-r shares g; r-z does not.
    var_color = sigma_first**2 + sigma_second**2
    covariance = sigma_g**2 if cov_g_color else 0.0
    expected = np.sqrt(np.maximum(0, var_color + slope**2*sigma_g**2
                                  - 2*slope*covariance))
    return resid, np.where(inside, expected, np.nan)


def audit_cluster(name, phase_cfg):
    cfg = SYSTEMS[name]
    old_path = Path("data")/f"cluster_rgb_{name}_v1"/"members.fits"
    old = fitsio.read(str(old_path))
    order = np.argsort(old["source_id"])
    old = old[order]
    rows, photo, raw_count = matched_sweeps(cfg, old["source_id"], prefer="gr")
    index = np.searchsorted(old["source_id"], rows["REF_ID"])
    if np.any(index >= len(old)) or np.any(old["source_id"][index] != rows["REF_ID"]):
        raise ValueError(f"Member identity mismatch in {name}")
    original = old[index]
    mu = cfg["mu"]
    phase = classify(original["gaia_g"]-mu, original["bp_rp"], phase_cfg[name])
    table = structured({
        "source_id": rows["REF_ID"].astype(np.int64), "ls_id": legacy_ids(rows),
        "original_ls_id": original["ls_id"], "release": rows["RELEASE"],
        "phase": phase, "pmember": original["pmember"],
        "radius_arcmin": original["radius_arcmin"], "ebv": rows["EBV"],
        "g0": photo["g0"], "r0": photo["r0"], "z0": photo["z0"],
        "gr0": photo["gr0"], "rz0": photo["rz0"],
        "sigma_g": photo["sigma_g"], "sigma_r": photo["sigma_r"],
        "sigma_z": photo["sigma_z"], "snr_g": photo["snr_g"],
        "snr_r": photo["snr_r"], "snr_z": photo["snr_z"],
        "gr_base": photo["gr_base"], "gr_clean": photo["gr_clean"],
        "grz_clean_same_row": photo["grz_clean"],
        "frozen_grz_clean": original["grz_clean"],
        "maskbits": rows["MASKBITS"],
        **{key.lower(): rows[key] for key in rows.dtype.names
           if key.startswith(("FRACMASKED_", "FRACIN_", "FRACFLUX_", "RCHISQ_"))
           and key.endswith(("_G", "_R", "_Z"))},
    })
    out = Path("data/cluster_rgb_gr_v2")/f"{name}.fits"
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(out), table)
    return table, original, summarize_cluster(name, table, raw_count)


def summarize_cluster(name, table, raw_count):
    cfg = SYSTEMS[name]
    mu = cfg["mu"]
    phase = table["phase"].astype("U")
    old_path = Path("data")/f"cluster_rgb_{name}_v1"/"members.fits"
    out = Path("data/cluster_rgb_gr_v2")/f"{name}.fits"
    high = table["pmember"] > .9
    counts = {}
    for group in ("upper_rgb", "lower_rgb", "hb", "possible_agb"):
        sel = high & (phase == group)
        gr = sel & table["gr_clean"]
        same_z = gr & ~table["grz_clean_same_row"]
        counts[group] = {
            "members": int(sel.sum()), "gr_base": int((sel & table["gr_base"]).sum()),
            "gr_clean": int(gr.sum()),
            "grz_clean_same_row": int((sel & table["grz_clean_same_row"]).sum()),
            "grz_clean_frozen_row": int((sel & table["frozen_grz_clean"]).sum()),
            "gr_clean_z_only_failure_same_row": int(same_z.sum()),
            "gr_clean_original_row_changed": int((gr & (table["ls_id"] != table["original_ls_id"])).sum()),
            "mg_gr": describe(table["g0"][gr]-mu),
            "gr0_gr": describe(table["gr0"][gr]),
            "radius_gr": describe(table["radius_arcmin"][gr]),
            "radius_z_only": describe(table["radius_arcmin"][same_z]),
            "fracmasked_g_gr": describe(table["fracmasked_g"][gr]),
            "fracmasked_r_gr": describe(table["fracmasked_r"][gr]),
        }
    release, nrelease = np.unique(table["release"], return_counts=True)
    info = {"system": name, "natural_system": SYSTEM_LABEL[name],
            "input_v1": str(old_path), "v2_data": str(out),
            "raw_matching_rows": raw_count, "unique_members": len(table),
            "release_counts": {str(k): int(v) for k,v in zip(release,nrelease)},
            "quality": "gr: same existing per-band cuts on g,r only; grz is retained separately",
            "phase_counts": counts,
            "shared_mu_mag": cfg["mu"], "shared_mu_stat": cfg["mu_stat"],
            "shared_mu_sys": cfg["mu_sys"]}
    return info


def core_loci(name, new, old):
    mu = SYSTEMS[name]["mu"]
    phase = new["phase"].astype("U")
    stage = np.isin(phase, CORE_PHASES[name])
    high = new["pmember"] > .9
    core_a = high & stage & new["gr_clean"] & np.isfinite(new["g0"]+new["gr0"])
    # B is precisely the frozen grz-selected row, including its photometry.
    core_b = high & stage & old["grz_clean"] & np.isfinite(old["g0"]+old["gr0"]+old["rz0"])
    out = {}
    for band, data, mask, field in (("gr", new, core_a, "gr0"),
                                    ("grz", old, core_b, "gr0"),
                                    ("grz_rz", old, core_b, "rz0")):
        groups = {}
        for p in CORE_PHASES[name]:
            use = mask & (phase == p)
            groups[p] = {"n": int(use.sum()),
                         "mg": describe(data["g0"][use]-mu),
                         "points": sequence(data["g0"][use]-mu, data[field][use])}
        use = mask
        out[band] = {"core_n": int(use.sum()), "stages": groups,
                     "points": sequence(data["g0"][use]-mu, data[field][use])}
    # Separate exclusions for the small M3 upper and NGC 2419 red-HB overlap.
    for p in ("upper_rgb", "lower_rgb"):
        if p not in CORE_PHASES[name]:
            use = high & (phase == p) & new["gr_clean"]
            out[f"diagnostic_{p}"] = {"n": int(use.sum()),
                "gr_points": sequence(new["g0"][use]-mu, new["gr0"][use], minimum=3)}
    return out


def residual_audit(name, tab, loci):
    mu = SYSTEMS[name]["mu"]
    use = ((tab["pmember"] > .9) & tab["gr_clean"]
           & np.isin(tab["phase"].astype("U"), CORE_PHASES[name]))
    a = tab[use]
    mg = a["g0"].astype(float)-mu
    gr = a["gr0"].astype(float)
    sg, sr = a["sigma_g"].astype(float), a["sigma_r"].astype(float)
    points = loci["gr"]["points"]
    resid, expect = local_residuals(mg, gr, sg, sg, sr, points, True)
    result = {"n_inside_curve": int(np.isfinite(resid).sum()),
              "local_color_residual": describe(resid),
              "propagated_photometric_sigma": describe(expect),
              "standardized_residual": describe(resid/np.maximum(expect, 1e-5)),
              "radius_halves": {}, "magnitude_halves": {}, "fracmasked_halves": {}}
    for key, values in (("radius_halves", a["radius_arcmin"]),
                        ("magnitude_halves", mg),
                        ("fracmasked_halves", a["fracmasked_g"]+a["fracmasked_r"])):
        split = np.nanmedian(values)
        result[key] = {"split": float(split),
                       "low": describe(resid[values <= split]),
                       "high": describe(resid[values > split])}
    return result


def rz_residual_audit(name, new, old, loci):
    mu = SYSTEMS[name]["mu"]
    use = ((new["pmember"] > .9) & old["grz_clean"]
           & np.isin(new["phase"].astype("U"), CORE_PHASES[name]))
    a = old[use]
    mg = a["g0"].astype(float)-mu
    sg = a["sigma_g"].astype(float)
    sr = a["sigma_r"].astype(float)
    sz = a["sigma_z"].astype(float)
    resid, expect = local_residuals(mg, a["rz0"].astype(float),
                                    sg, sr, sz,
                                    loci["grz_rz"]["points"], False)
    return {"n_inside_curve": int(np.isfinite(resid).sum()),
            "local_color_residual": describe(resid),
            "propagated_photometric_sigma": describe(expect),
            "standardized_residual": describe(resid/np.maximum(expect, 1e-5))}


def decam_overlap(loci):
    out = {}
    for band in ("gr", "grz_rz"):
        a = loci["m3"][band]["stages"]["lower_rgb"]["points"]
        b = loci["m53"][band]["stages"]["lower_rgb"]["points"]
        if len(a) < 2 or len(b) < 2:
            out[band] = {"n": 0}
            continue
        xa, ya = np.array([p["mg"] for p in a]), np.array([p["color"] for p in a])
        xb, yb = np.array([p["mg"] for p in b]), np.array([p["color"] for p in b])
        low, high = max(xa[0],xb[0]), min(xa[-1],xb[-1])
        grid = np.arange(np.ceil(low*4)/4, high+1e-8, .25)
        delta = np.interp(grid,xa,ya)-np.interp(grid,xb,yb)
        dust_color = RG-RR if band == "gr" else RR-RZ
        common_scale = {}
        for name, x, y, sign in (("m3",xa,ya,1), ("m53",xb,yb,-1)):
            cfg = SYSTEMS[name]
            sigma_mu = float(np.hypot(cfg["mu_stat"],cfg["mu_sys"]))
            trials = {}
            for kind, amount in (("mu",sigma_mu),
                                 ("dust",DUST_SCALE_TRIAL[name])):
                effects = []
                for direction in (-1,1):
                    dmu = direction*amount if kind == "mu" else 0
                    debv = direction*amount if kind == "dust" else 0
                    changed = np.interp(grid,x-dmu-RG*debv,y-dust_color*debv)
                    effects.append(float(np.median(sign*(changed-np.interp(grid,x,y)))))
                trials[kind] = {"assumed_sigma": amount,
                                "median_color_difference_shift": effects}
            common_scale[name] = trials
        out[band] = {"n": len(grid), "mg": grid.tolist(),
                     "m3_minus_m53_color": delta.tolist(), "summary": describe(delta),
                     "whole_cluster_scale_sensitivity": common_scale}
    return out


def plot_loci(path, loci):
    fig, axes = plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    specs = (("m3","M3 lower RGB","tab:blue"),
             ("m53","M53 upper+lower RGB","tab:orange"),
             ("ngc2419","NGC 2419 upper RGB","tab:green"))
    for name,label,color in specs:
        for ax,band in zip(axes,("gr","grz_rz")):
            pts = loci[name][band]["points"]
            if pts:
                ax.plot([p["color"] for p in pts], [p["mg"] for p in pts],
                        "o-", color=color, ms=3, label=f"{label} (n={loci[name][band]['core_n']})")
                for p in pts:
                    ax.plot([p["p16"],p["p84"]],[p["mg"]]*2,
                            color=color, alpha=.5, lw=1.5)
    for name, phase, label, color in (
            ("m3", "diagnostic_upper_rgb", "M3 upper edge (diagnostic)", "tab:red"),
            ("ngc2419", "diagnostic_lower_rgb",
             "NGC 2419 lower/RHB overlap (diagnostic)", "0.45")):
        diagnostic = loci[name][phase]
        points = diagnostic["gr_points"]
        if points:
            axes[0].plot([p["color"] for p in points],
                         [p["mg"] for p in points], "o--", color=color,
                         ms=3, lw=1, alpha=.8,
                         label=f"{label} (n={diagnostic['n']})")
    axes[0].set(xlabel="(g−r)₀",ylabel="g₀ − cluster μ")
    axes[1].set(xlabel="(r−z)₀",ylabel="g₀ − cluster μ")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Stage-matched native-system loci; dashed: separate diagnostics; bars: within-bin 16–84%")
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=170)
    plt.close(fig)


def main():
    require_compute()
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-from-cache", action="store_true")
    args = parser.parse_args()
    manifest = Path("manifests/cluster_rgb_system_audit_v2.json")
    if manifest.exists() and not args.recompute_from_cache:
        raise FileExistsError("manifests/cluster_rgb_system_audit_v2.json")
    phase_cfg = json.loads(CONFIG.read_text())["systems"]
    tables, originals, inventories, loci, residuals = {}, {}, {}, {}, {}
    for name in SYSTEMS:
        if args.recompute_from_cache:
            tab = fitsio.read(str(Path("data/cluster_rgb_gr_v2")/f"{name}.fits"))
            old = fitsio.read(str(Path("data")/f"cluster_rgb_{name}_v1"/"members.fits"))
            old = old[np.argsort(old["source_id"])]
            idx = np.searchsorted(old["source_id"],tab["source_id"])
            if np.any(idx >= len(old)) or np.any(old["source_id"][idx] != tab["source_id"]):
                raise ValueError(f"Cached member identity mismatch in {name}")
            old = old[idx]
            previous = json.loads((Path("manifests")/
                f"cluster_rgb_{name}_inventory_v1.json").read_text())
            info = summarize_cluster(name, tab,
                                     raw_count=previous["counts"]["dr11_matching_rows"])
        else:
            tab, old, info = audit_cluster(name, phase_cfg)
        tables[name], originals[name], inventories[name] = tab, old, info
        if name in CORE_PHASES:
            loci[name] = core_loci(name, tab, old)
            residuals[name] = {"gr": residual_audit(name, tab, loci[name]),
                               "rz": rz_residual_audit(name, tab, old, loci[name])}
    fig = Path("figures/cluster_rgb/04_stage_matched_native_loci_v2.png")
    plot_loci(fig,loci)
    result = {"method": "Full v1 member parent rejoined to DR11; gr row selected by gr quality; frozen grz row retained separately",
              "source": "DR11 sweep RELEASE; north BASS/MzLS, south DECam",
              "systems": inventories, "phase_aligned_loci": loci,
              "local_color_residuals": residuals,
              "decam_m3_m53_same_mg": decam_overlap(loci),
              "figure": str(fig), "compute_host": socket.gethostname()}
    if args.recompute_from_cache and manifest.exists():
        shutil.copy2(manifest, Path("data/cluster_rgb_gr_v2/pre_stage_fix_manifest.json"))
    write_json(manifest,result)
    print(json.dumps({"counts": {n:v["phase_counts"] for n,v in inventories.items()},
                      "decam_overlap": result["decam_m3_m53_same_mg"]}),flush=True)


if __name__ == "__main__":
    main()

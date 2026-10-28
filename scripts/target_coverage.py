#!/usr/bin/env python3
"""Measure application input layers and a fixed color-only cold candidate rule."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from photdis.audit import checked_gaia_color
from photdis.followup import common_nonoptical, identity_hash
from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.stage1 import require_compute, write_json

OUT = Path("data/target_coverage_v1")
FIG = Path("figures/target_coverage")
FIELD_FILES = {"original_ra000_010_dec_m010_000": "data/pilot_10deg_v1/field_diagnostics.fits",
               "cal_ra140_150_dec_p005_p015": "data/field_cal_ra140_150_v1/field_diagnostics.fits",
               "test_ra190_200_dec_p015_p025": "data/field_test_ra190_200_v1/field_diagnostics.fits",
               "test_ra320_330_dec_m025_m015": "data/field_test_ra320_330_v1/field_diagnostics.fits"}
COLS = ["source_id", "ra", "dec", "g0", "p_corr", "gr0", "rz0", "grz_base",
        "grz_w1", "grz_w1_w2", "grz_high_quality", "anchor_astrometry",
        "contaminant_candidate", "desi_quality", "desi_match_ambiguous", "desi_logg",
        "desi_teff", "desi_known_contaminant", "apogee_quality", "apogee_match_ambiguous",
        "apogee_logg", "apogee_known_contaminant", "gaia_gaia_match",
        "gaia_phot_bp_mean_mag", "gaia_phot_rp_mean_mag", "gaia_phot_bp_mean_flux",
        "gaia_phot_rp_mean_flux", "gaia_phot_bp_mean_flux_error",
        "gaia_phot_rp_mean_flux_error", "gaia_phot_bp_n_obs", "gaia_phot_rp_n_obs",
        "gaia_bp_rp"]


def fixed_rule():
    table = fitsio.read("data/pilot_audit_v1/reference.fits")
    external = annotate_holdouts(load_reference_catalogs("data/reference"),
                                 table["source_id"], table["gaia_ra"], table["gaia_dec"])["external_any"]
    eligible, _, _ = common_nonoptical(table, external)
    strict = table["anchor"] & eligible
    baseline = json.loads(Path("manifests/cold_baseline.json").read_text())
    cells = hp.ang2pix(64, table["gaia_ra"], table["gaia_dec"], nest=True, lonlat=True)
    split = np.array([baseline["split"]["cell_map"].get(str(int(c)), 3) for c in cells])
    fit = strict & (split != 2)
    color = np.column_stack((table["gr0"][fit], table["rz0"][fit]))
    low = np.quantile(color, .01, axis=0)
    high = np.quantile(color, .99, axis=0)
    scale = np.array([.2, .15])
    tree = cKDTree(color / scale)
    nearest = tree.query(color / scale, k=10)[0][:, -1]
    threshold = float(np.quantile(nearest, .95))
    return {"fit_rows": int(fit.sum()), "fit_identity_sha256": identity_hash(table["source_id"][fit]),
            "color_1pct": low.tolist(), "color_99pct": high.tolist(),
            "support_scale_mag": scale.tolist(), "support_10th_neighbor_95pct": threshold,
            "rule": "gr0 and rz0 within fit-star 1-99% boxes; support is separate 10th-neighbor radius"}, tree


def breakdown(mask, reference, selected):
    total = int(mask.sum())
    return {"n": total, "selected": int((mask & selected).sum()),
            "fraction_selected": float((mask & selected).sum() / total) if total else None}


def field_result(name, path, rule, tree):
    table = fitsio.read(path, columns=COLS)
    base = table["grz_base"] & np.isfinite(table["gr0"]) & np.isfinite(table["rz0"])
    colors = np.column_stack((table["gr0"], table["rz0"]))
    low, high = np.array(rule["color_1pct"]), np.array(rule["color_99pct"])
    candidate = base & np.all((colors >= low) & (colors <= high), axis=1)
    support = np.zeros(len(table), bool)
    good = np.flatnonzero(base)
    distance = tree.query(colors[good] / rule["support_scale_mag"], k=10)[0][:, -1]
    support[good] = distance <= rule["support_10th_neighbor_95pct"]
    qualified = candidate & support
    gaia = {key[5:]: table[key] for key in COLS if key.startswith("gaia_")}
    checked = checked_gaia_color(gaia)
    mg = table["g0"] - 10 + 5 * np.log10(table["p_corr"])
    gaia_reference = (base & table["anchor_astrometry"] & checked["bp_rp_precise"] &
                      ~table["contaminant_candidate"] & np.isfinite(mg))
    cold = (gaia_reference & (checked["bp_rp_checked"] >= 1.3) &
            (checked["bp_rp_checked"] < 3.5) & (mg > 5))
    other = gaia_reference & ~cold
    spectro = (base & table["desi_quality"] & ~table["desi_match_ambiguous"] &
               np.isfinite(table["desi_logg"]) & np.isfinite(table["desi_teff"]))
    cool = spectro & (table["desi_teff"] >= 3000) & (table["desi_teff"] < 5000)
    dwarf = cool & (table["desi_logg"] >= 3.5)
    giant = cool & (table["desi_logg"] < 3.5)
    return {"catalog_rows": len(table), "input_layers": {
                key: int(table[key].sum()) for key in ("grz_base", "grz_w1", "grz_w1_w2", "grz_high_quality")},
            "color_candidate": int(candidate.sum()), "color_support": int(support.sum()),
            "candidate_and_support": int(qualified.sum()),
            "Gaia_reference": {"cold_dwarf": breakdown(cold, gaia_reference, candidate),
                               "other": breakdown(other, gaia_reference, candidate),
                               "cold_with_support": breakdown(cold, gaia_reference, qualified),
                               "other_with_support": breakdown(other, gaia_reference, qualified),
                               "candidate_purity_cold": float((candidate & cold).sum() / (candidate & gaia_reference).sum())
                               if (candidate & gaia_reference).any() else None},
            "DESI_quality_cool": {"dwarf": breakdown(dwarf, cool, candidate),
                                   "giant": breakdown(giant, cool, candidate),
                                   "dwarf_with_support": breakdown(dwarf, cool, qualified),
                                   "giant_with_support": breakdown(giant, cool, qualified),
                                   "candidate_purity_dwarf": float((candidate & dwarf).sum() / (candidate & cool).sum())
                                   if (candidate & cool).any() else None}}


def main():
    require_compute()
    if (OUT / "summary.json").exists():
        raise FileExistsError(OUT / "summary.json")
    rule, tree = fixed_rule()
    summary = {"fixed_from_original_non_test_cold_anchors": rule,
               "fields": {}, "compute_host": socket.gethostname()}
    for name, path in FIELD_FILES.items():
        summary["fields"][name] = field_result(name, path, rule, tree)
        print(json.dumps({"field": name, **summary["fields"][name]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fields = list(FIELD_FILES)
    labels = ["pilot", "cal", "north test", "south test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(fields))
    for key, color in (("grz_base", "#3a79a3"), ("grz_w1", "#56904e"),
                       ("grz_w1_w2", "#bb7340")):
        axes[0].plot(x, [summary["fields"][f]["input_layers"][key] /
                          summary["fields"][f]["catalog_rows"] for f in fields],
                     "o-", label=key, color=color)
    for label, key, color in (("Gaia cold", "cold_dwarf", "#3a79a3"),
                               ("Gaia other", "other", "#bb7340")):
        axes[1].plot(x, [summary["fields"][f]["Gaia_reference"][key]["fraction_selected"]
                          for f in fields], "o-", label=label, color=color)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Fraction of stored pilot rows")
    axes[1].set_ylabel("Color-only candidate selection fraction")
    fig.tight_layout()
    fig.savefig(FIG / "01_inputs_and_cold_candidate.png", dpi=170)
    plt.close(fig)
    write_json(OUT / "summary.json", summary)
    write_json("manifests/target_coverage.json", summary)


if __name__ == "__main__":
    main()

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

from photdis.audit import checked_gaia_color
from photdis.followup import common_nonoptical, identity_hash
from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.stage1 import require_compute, write_json
from photdis.support import support_radius, supported

OUT = Path("data/target_coverage_v2")
FIG = Path("figures/target_coverage_v2")
FIELD_FILES = {"cal_ra140_150_dec_p005_p015": "data/field_cal_ra140_150_v1/field_diagnostics.fits",
               "test_ra190_200_dec_p015_p025": "data/field_test_ra190_200_v1/field_diagnostics.fits",
               "test_ra320_330_dec_m025_m015": "data/field_test_ra320_330_v1/field_diagnostics.fits"}
FIELD_MANIFESTS = {"cal_ra140_150_dec_p005_p015": "manifests/field_cal_ra140_150.json",
                   "test_ra190_200_dec_p015_p025": "manifests/field_test_ra190_200.json",
                   "test_ra320_330_dec_m025_m015": "manifests/field_test_ra320_330.json"}
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
    tree, threshold = support_radius(color, scale, neighbors=10, quantile=.95)
    return {"fit_rows": int(fit.sum()), "fit_identity_sha256": identity_hash(table["source_id"][fit]),
            "color_1pct": low.tolist(), "color_99pct": high.tolist(),
            "support_scale_mag": scale.tolist(), "support_10th_neighbor_95pct": threshold,
            "rule": "gr0 and rz0 within fit-star 1-99% boxes; support uses distance to the tenth other training star"}, tree


def breakdown(mask, selected):
    total = int(mask.sum())
    return {"n": total, "selected": int((mask & selected).sum()),
            "fraction_selected": float((mask & selected).sum() / total) if total else None}


def field_result(name, path, rule, tree):
    table = fitsio.read(path, columns=COLS)
    domain = json.loads(Path(FIELD_MANIFESTS[name]).read_text())["domain_psf_rows"]
    base = table["grz_base"] & np.isfinite(table["gr0"]) & np.isfinite(table["rz0"])
    colors = np.column_stack((table["gr0"], table["rz0"]))
    low, high = np.array(rule["color_1pct"]), np.array(rule["color_99pct"])
    candidate = base & np.all((colors >= low) & (colors <= high), axis=1)
    support = np.zeros(len(table), bool)
    good = np.flatnonzero(base)
    support[good] = supported(tree, colors[good], rule["support_scale_mag"],
                              rule["support_10th_neighbor_95pct"], neighbors=10)
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
    nonstar = base & table["desi_known_contaminant"] & ~table["desi_match_ambiguous"]
    return {"domain_psf_rows": domain, "stored_diagnostic_rows": len(table), "input_layers": {
                key: int(table[key].sum()) for key in ("grz_base", "grz_w1", "grz_w1_w2", "grz_high_quality")},
            "color_candidate": int(candidate.sum()), "color_support": int(support.sum()),
            "candidate_and_support": int(qualified.sum()),
            "Gaia_reference": {"cold_dwarf": breakdown(cold, candidate),
                               "other": breakdown(other, candidate),
                               "cold_with_support": breakdown(cold, qualified),
                               "other_with_support": breakdown(other, qualified)},
            "DESI_quality_cool": {"dwarf": breakdown(dwarf, candidate),
                                   "giant": breakdown(giant, candidate),
                                   "dwarf_with_support": breakdown(dwarf, qualified),
                                   "giant_with_support": breakdown(giant, qualified)},
            "DESI_known_nonstar": breakdown(nonstar, candidate),
            "selection_rates": {
                "Gaia_cold_color_box_recall": breakdown(cold, candidate)["fraction_selected"],
                "Gaia_cold_color_and_support_recall": breakdown(cold, qualified)["fraction_selected"],
                "DESI_cool_dwarf_color_box_recall": breakdown(dwarf, candidate)["fraction_selected"],
                "DESI_cool_dwarf_color_and_support_recall": breakdown(dwarf, qualified)["fraction_selected"],
                "DESI_known_nonstar_color_box_leakage": breakdown(nonstar, candidate)["fraction_selected"],
                "DESI_known_nonstar_color_and_support_leakage": breakdown(nonstar, qualified)["fraction_selected"]}}


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
    labels = ["cal", "north test", "south test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(fields))
    for key, color in (("grz_base", "#3a79a3"), ("grz_w1", "#56904e"),
                       ("grz_w1_w2", "#bb7340")):
        axes[0].plot(x, [summary["fields"][f]["input_layers"][key] /
                          summary["fields"][f]["domain_psf_rows"] for f in fields],
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
    axes[0].set_ylabel("Fraction of field PSF rows")
    axes[1].set_ylabel("Color-only candidate selection fraction")
    fig.tight_layout()
    fig.savefig(FIG / "01_inputs_and_cold_candidate.png", dpi=170)
    plt.close(fig)
    write_json(OUT / "summary.json", summary)
    write_json("manifests/target_coverage_v2.json", summary)


if __name__ == "__main__":
    main()

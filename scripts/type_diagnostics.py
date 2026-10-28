#!/usr/bin/env python3
"""Audit conditional cold-branch selection by DESI label, color and brightness."""
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
from photdis.support import supported
from target_coverage import fixed_rule

OUT = Path("data/type_diagnostics_dev_v1")
FIG = Path("figures/type_diagnostics_dev_v1")
FIELDS = {
    "prior_cal": "data/field_cal_ra140_150_v1/field_diagnostics.fits",
    "prior_north": "data/field_test_ra190_200_v1/field_diagnostics.fits",
    "prior_south": "data/field_test_ra320_330_v1/field_diagnostics.fits",
    "new_cal": "data/field_dev_cal_ra205_215_v1/field_diagnostics.fits",
}
FINAL_FIELDS = {
    "final_ra230_240": "data/field_final_ra230_240_v1/field_diagnostics.fits",
    "final_ra050_060": "data/field_final_ra050_060_v1/field_diagnostics.fits",
}
COLS = ["grz_base", "grz_w1", "grz_w1_w2", "gr0", "rz0", "r",
        "desi_quality", "desi_match_ambiguous", "desi_known_contaminant",
        "desi_teff", "desi_logg"]
R_BINS = (14, 16, 18, 20, 22, 25)
RZ_BINS = (-1, .3, .6, .9, 1.2, 3)


def count(mask, candidate, qualified):
    n = int(mask.sum())
    box = int((mask & candidate).sum())
    both = int((mask & qualified).sum())
    def wilson(success):
        if not n:
            return None
        z = 1.96
        p = success / n
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        radius = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        return [float(center-radius), float(center+radius)]
    return {"n_labeled": n, "color_box_selected": box,
            "box_and_support_selected": both,
            "color_box_rate_given_label": box / n if n else None,
            "color_box_rate_wilson95": wilson(box),
            "box_and_support_rate_given_label": both / n if n else None,
            "box_and_support_rate_wilson95": wilson(both)}


def field_summary(path, rule, tree):
    table = fitsio.read(path, columns=COLS)
    base = table["grz_base"] & np.isfinite(table["gr0"]) & np.isfinite(table["rz0"])
    color = np.column_stack((table["gr0"], table["rz0"]))
    low = np.asarray(rule["color_1pct"])
    high = np.asarray(rule["color_99pct"])
    candidate = base & np.all((color >= low) & (color <= high), axis=1)
    support = np.zeros(len(table), bool)
    support[base] = supported(tree, color[base], rule["support_scale_mag"],
                              rule["support_10th_neighbor_95pct"])
    qualified = candidate & support
    reliable = table["desi_quality"] & ~table["desi_match_ambiguous"]
    nonstar = table["desi_known_contaminant"] & ~table["desi_match_ambiguous"]
    parameters = reliable & ~nonstar & np.isfinite(table["desi_teff"]) & np.isfinite(table["desi_logg"])
    cool = parameters & (table["desi_teff"] >= 3000) & (table["desi_teff"] < 5000)
    labels = {"DESI_cool_dwarf": cool & (table["desi_logg"] >= 3.5),
              "DESI_cool_giant": cool & (table["desi_logg"] < 3.5),
              "DESI_other_parameterized_star": parameters & ~cool,
              "DESI_known_nonstar": nonstar,
              "unverified_by_DESI": ~parameters & ~nonstar}
    result = {"stored_diagnostic_rows": len(table), "grz_base": int(base.sum()),
              "color_box_candidates": int(candidate.sum()),
              "color_and_support_candidates": int(qualified.sum()),
              "label_scope": "DESI quality, unambiguous crossmatch and finite Teff/logg for stars; unambiguous known nonstar classification; other rows are unverified. DESI targets are not a random sample of all photometric candidates.",
              "class_conditional_rates_not_candidate_purity": True,
              "labels": {name: count(mask, candidate, qualified)
                         for name, mask in labels.items()},
              "strata": {}}
    for axis, values, edges in (("observed_r", table["r"], R_BINS),
                                ("observed_r_minus_z", table["rz0"], RZ_BINS)):
        groups = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = np.isfinite(values) & (values >= lo) & (values < hi)
            groups.append({"bin": [lo, hi], "all_rows": int(inside.sum()),
                           "labels": {name: count(mask & inside, candidate, qualified)
                                      for name, mask in labels.items()}})
        result["strata"][axis] = groups
    result["candidate_input_tiers"] = {
        name: int((candidate & table[name]).sum())
        for name in ("grz_base", "grz_w1", "grz_w1_w2")}
    return result


def plot_summary(summary, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = ("DESI_cool_dwarf", "DESI_cool_giant", "DESI_known_nonstar")
    fields = list(summary["fields"])
    x = np.arange(len(fields), dtype=float)
    for index, name in enumerate(labels):
        points = [(j, summary["fields"][field]["labels"][name])
                  for j, field in enumerate(fields)]
        points = [(j, value) for j, value in points if value["n_labeled"]]
        if not points:
            continue
        px = np.array([j + (index - 1) * .10 for j, _ in points])
        y = np.array([value["color_box_rate_given_label"] for _, value in points])
        lower = np.array([value["color_box_rate_wilson95"][0] for _, value in points])
        upper = np.array([value["color_box_rate_wilson95"][1] for _, value in points])
        ax.errorbar(px, y, yerr=[y - lower, upper - y], fmt="o", capsize=3,
                    label=name.replace("DESI_", "").replace("_", " "))
        for xx, yy, (_, value) in zip(px, y, points):
            ax.annotate(f'n={value["n_labeled"]}', (xx, yy), xytext=(3, 4),
                        textcoords="offset points", fontsize=7)
    ax.set_xticks(x, fields)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(color-box selected | DESI label)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_class_conditional_box_rates.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    fields = FINAL_FIELDS if args.final else FIELDS
    out = Path("data/type_diagnostics_final_v1") if args.final else OUT
    figure_dir = Path("figures/type_diagnostics_final_v1") if args.final else FIG
    manifest = Path("manifests/type_diagnostics_final.json") if args.final else Path("manifests/type_diagnostics_dev.json")
    if args.plot_only:
        plot_summary(json.loads(manifest.read_text()), figure_dir)
        return
    require_compute()
    if (out / "summary.json").exists():
        raise FileExistsError(out / "summary.json")
    rule, tree = fixed_rule()
    summary = {"selection_rule": rule, "fields": {},
               "compute_host": socket.gethostname()}
    for name, path in fields.items():
        summary["fields"][name] = field_summary(path, rule, tree)
        print(json.dumps({"field": name,
                          "labels": summary["fields"][name]["labels"]}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    plot_summary(summary, figure_dir)
    write_json(out / "summary.json", summary)
    write_json(manifest, summary)


if __name__ == "__main__":
    main()

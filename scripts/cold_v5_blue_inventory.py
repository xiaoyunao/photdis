#!/usr/bin/env python3
"""Inventory the metal-poor blue side of the frozen cold-v4 color box."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import joblib
import numpy as np

from photdis.audit import rvs_metallicity
from photdis.holdouts import load_reference_catalogs
from photdis.stage1 import require_compute, write_json
from cold_v4 import CAL_CONFIG, FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED, sha
from cold_v4_type_audit import COLS, class_masks, unique_spectra
from cold_v5_xd_fit import load_training

RESULT = Path("data/cold_v5_blue/inventory.json")
SUMMARY = Path("manifests/cold_v5_blue_inventory.json")
READ = sorted(set(COLS + ["anchor", "g0", "mg", "p_corr", "snr_corr", "ebv",
                          "gaia_bp_rp", "grz_fraction_clean", "grz_mask_clean"]))
R_BINS = ((14, 16), (16, 18), (18, 20), (20, 22), (22, 26))
GR_BINS = ((0., .6), (.6, .909), (.909, 1.2), (1.2, 1.6))


def _count(mask):
    return int(np.sum(mask))


def _bin_counts(mask, values, bins):
    return [{"range": [lo, hi], "count": _count(mask &
            (values >= lo) & (values < hi))} for lo, hi in bins]


def _records(table, indices, field, limit=600):
    if len(indices) > limit:
        indices = np.sort(np.random.default_rng(20261029).choice(
            indices, limit, replace=False))
    names = ("ls_id", "source_id", "ra", "dec", "r", "gr0", "rz0",
             "mg", "p_corr", "snr_corr", "gaia_bp_rp", "desi_teff",
             "desi_logg", "desi_feh", "desi_feh_err")
    result = []
    for i in indices:
        item = {"field": field}
        for name in names:
            value = table[name][i]
            if name in ("ls_id", "source_id"):
                item[name] = int(value)
            else:
                value = float(value)
                item[name] = value if np.isfinite(value) else None
        result.append(item)
    return result


def field_inventory(config, color_low):
    task = json.loads(Path(config).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    raw = fitsio.read(str(path), columns=READ)
    table = {name: raw[name] for name in raw.dtype.names}
    unique, dedup = unique_spectra(table)
    classes = class_masks(table, unique & table["grz_base"])
    calibrated, valid = rvs_metallicity(table["desi_feh"],
                                        table["desi_teff"], table["desi_logg"])
    abundance = (classes["cool_dwarf"] & valid &
                 np.isfinite(table["desi_feh_err"]) &
                 (table["desi_feh_err"] > 0))
    poor = abundance & (calibrated < -1.5)
    blue = np.isfinite(table["gr0"]) & (table["gr0"] < color_low)
    low_dust = np.isfinite(table["ebv"]) & (table["ebv"] < .03)
    blue_clean = blue & table["grz_clean"] & low_dust
    gaia = (np.asarray(table["anchor"], bool) & blue_clean &
            np.isfinite(table["mg"]) & np.isfinite(table["snr_corr"]) &
            (table["snr_corr"] >= 20))
    labels = {**classes, "calibrated_FeH_lt_minus_1p5_dwarf": poor,
              "calibrated_domain_cool_dwarf": abundance}
    summary = {"field": task["name"], "input_rows": len(raw),
               "deduplication": dedup,
               "blue_definition": f"observed dereddened gr0 < frozen optical lower edge {color_low:.6f}; diagnostic region only",
               "blue_low_dust_grz_clean": _count(blue_clean),
               "blue_strict_Gaia_anchors": _count(gaia),
               "blue_strict_Gaia_by_r": _bin_counts(gaia, table["r"], R_BINS),
               "labels": {}, "plot_samples": {}}
    for name, mask in labels.items():
        subset = mask & blue
        summary["labels"][name] = {
            "qualified_all_colors": _count(mask),
            "blue": _count(subset),
            "blue_grz_clean_low_dust": _count(subset & blue_clean),
            "blue_gaia_anchor_overlap": _count(subset & gaia),
            "blue_below_rz_lower_edge": _count(subset &
                (table["rz0"] < .401)),
            "blue_by_r": _bin_counts(subset, table["r"], R_BINS),
            "all_by_gr": _bin_counts(mask, table["gr0"], GR_BINS)}
    summary["plot_samples"]["gaia_blue"] = _records(table,
        np.flatnonzero(gaia), task["name"], limit=1500)
    for name in ("calibrated_FeH_lt_minus_1p5_dwarf", "cool_dwarf",
                 "cool_giant", "known_QSO_or_galaxy"):
        summary["plot_samples"][name] = _records(table,
            np.flatnonzero(labels[name] & blue), task["name"])
    return summary


def training_blue_coverage(references, color_low):
    result = {}
    for table, selected, _, meta in load_training(references):
        blue = (selected & table["grz_base"] & np.isfinite(table["gr0"]) &
                (table["gr0"] < color_low))
        result[meta["name"]] = {
            "cold_strict_anchor_rows": _count(selected),
            "blue_below_frozen_gr_lower_edge": _count(blue),
            "blue_clean_low_dust": _count(blue & table["grz_clean"] &
                (table["ebv"] < .03)),
            "blue_by_r": _bin_counts(blue, table["r"], R_BINS)}
    return result


def main():
    require_compute()
    if RESULT.exists() or SUMMARY.exists():
        raise FileExistsError("Blue inventory already exists")
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen preparation changed")
    prepared = json.loads(PREPARED.read_text())
    route_path = OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen route changed")
    route = joblib.load(route_path)
    low = float(route["optical_low"][0])
    references = load_reference_catalogs("data/reference")
    report = {"prepared_sha256": sha(PREPARED),
              "frozen_gr_lower_edge": low,
              "selection_semantics": "DESI classes conditional on unique quality spectra and grz_base; Gaia anchors are parallax/photometry-quality candidates, not metal-poor labels",
              "training": training_blue_coverage(references, low),
              "fields": {}, "compute_host": socket.gethostname()}
    for config in (CAL_CONFIG, *FINAL_CONFIGS):
        field = field_inventory(config, low)
        report["fields"][field["field"]] = field
        print(json.dumps({"field": field["field"],
              "poor_blue": field["labels"]["calibrated_FeH_lt_minus_1p5_dwarf"]["blue"],
              "gaia_blue": field["blue_strict_Gaia_anchors"]}), flush=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, report)
    summary = {**report, "fields": {name: {key: value for key, value in field.items()
        if key != "plot_samples"} for name, field in report["fields"].items()},
        "full_plot_data": str(RESULT), "full_plot_data_sha256": sha(RESULT)}
    write_json(SUMMARY, summary)


if __name__ == "__main__":
    main()

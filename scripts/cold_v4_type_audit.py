#!/usr/bin/env python3
"""Class-conditional DESI checks of the frozen cold-v4 photometric route."""
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
from photdis.route import FEATURES, LAYER, LEVEL, route_rows
from photdis.stage1 import require_compute, write_json
from cold_v4 import CAL_CONFIG, FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED, registration, sha

RESULT = Path("data/cold_v4_type/audit.json")
R_BINS = ((14, 16), (16, 18), (18, 20), (20, 22), (22, 26))
RZ_BINS = ((0, .6), (.6, 1.0), (1.0, 1.6), (1.6, 3.5))
ERROR_BINS = ((0, .03), (.03, .1), (.1, .2), (.2, float("inf")))
COLS = sorted(set((*FEATURES["C_grzW1W2"], *LAYER.values(),
                   "ls_id", "source_id", "ra", "dec", "r", "grz_clean",
                   "grz_high_quality", "grz_w1_high_quality", "full_high_quality",
                   "sigma_g", "sigma_r", "sigma_z", "sigma_w1", "sigma_w2",
                   "desi_match", "desi_match_ambiguous", "desi_object_id",
                   "desi_source_id", "desi_snr", "desi_separation_arcsec",
                   "desi_quality", "desi_is_stellar", "desi_known_contaminant",
                   "desi_teff", "desi_logg", "desi_feh", "desi_feh_err")))


def unique_spectra(table):
    """Choose one Legacy row per unambiguous DESI spectrum deterministically."""
    ids = np.char.strip(np.asarray(table["desi_object_id"]).astype("U"))
    matched = (np.asarray(table["desi_match"], bool) &
               ~np.asarray(table["desi_match_ambiguous"], bool) & (ids != ""))
    rows = np.flatnonzero(matched)
    if not len(rows):
        return np.zeros(len(ids), bool), {"matched_unambiguous_rows": 0,
                                           "unique_spectra": 0}
    snr = np.nan_to_num(np.asarray(table["desi_snr"][rows], float), nan=-np.inf)
    separation = np.nan_to_num(np.asarray(table["desi_separation_arcsec"][rows], float),
                               nan=np.inf)
    order = np.lexsort((np.asarray(table["ls_id"][rows]), separation, -snr, ids[rows]))
    sorted_rows = rows[order]
    _, first = np.unique(ids[sorted_rows], return_index=True)
    keep = np.zeros(len(ids), bool)
    keep[sorted_rows[first]] = True
    return keep, {"matched_unambiguous_rows": len(rows),
                  "unique_spectra": int(keep.sum()),
                  "duplicate_Legacy_rows_removed": int(len(rows) - keep.sum())}


def class_masks(table, unique):
    nonstar = unique & np.asarray(table["desi_known_contaminant"], bool)
    parameterized = (unique & table["desi_quality"] & table["desi_is_stellar"] &
                     ~nonstar & np.isfinite(table["desi_teff"]) &
                     np.isfinite(table["desi_logg"]))
    cool = parameterized & (table["desi_teff"] >= 3000) & (table["desi_teff"] < 5000)
    dwarf = cool & (table["desi_logg"] >= 3.5)
    giant = cool & (table["desi_logg"] < 3.5)
    return {"cool_dwarf": dwarf, "cool_giant": giant,
            "other_parameterized_star": parameterized & ~cool,
            "known_QSO_or_galaxy": nonstar,
            "other_or_unverified_spectrum": unique & ~parameterized & ~nonstar}


def used_error(table, tier):
    bands = ("g", "r", "z") + (("w1",) if LEVEL[tier] >= 2 else ()) + (("w2",) if LEVEL[tier] == 3 else ())
    return np.maximum.reduce([np.asarray(table[f"sigma_{band}"], float) for band in bands])


def wilson(success, total):
    if not total:
        return None
    z = 1.96
    rate = success / total
    center = (rate + z*z/(2*total)) / (1 + z*z/total)
    radius = z * np.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [float(center-radius), float(center+radius)]


def class_result(mask, table, routed):
    n = int(mask.sum())
    color = mask & routed["color_box"]
    selected = mask & routed["selected"]
    clean = selected & table["grz_clean"]
    result = {"n_labeled_in_grz_base": n,
              "color_box": int(color.sum()),
              "complete_route": int(selected.sum()),
              "complete_route_and_clean": int(clean.sum()),
              "route_rate_given_label": float(selected.sum()/n) if n else None,
              "route_rate_wilson95": wilson(int(selected.sum()), n),
              "clean_route_rate_given_label": float(clean.sum()/n) if n else None,
              "clean_route_rate_wilson95": wilson(int(clean.sum()), n),
              "by_tier": {}, "fallback_after_unsupported_higher_layer": {},
              "strata": {}}
    for tier, level in LEVEL.items():
        take = selected & (routed["tier"] == level)
        higher = [name for name in FEATURES if LEVEL[name] > level]
        unsupported = np.logical_or.reduce([
            table[LAYER[name]] & ~routed["support_by_tier"][name]
            for name in higher]) if higher else np.zeros(len(mask), bool)
        result["by_tier"][tier] = {"all": int(take.sum()),
                                  "grz_clean": int(np.sum(take & table["grz_clean"])),
                                  "grz_high_quality": int(np.sum(take & table["grz_high_quality"])),
                                  "used_tier_high_quality": int(np.sum(take & table[
                                      {"A_grz": "grz_high_quality",
                                       "B_grzW1": "grz_w1_high_quality",
                                       "C_grzW1W2": "full_high_quality"}[tier]]))}
        result["fallback_after_unsupported_higher_layer"][tier] = int(np.sum(take & unsupported))
    for name, values, bins in (("r", table["r"], R_BINS),
                               ("r_minus_z", table["rz0"], RZ_BINS)):
        result["strata"][name] = [
            {"bin": [lo, hi], "n_labeled": int(np.sum(mask & (values >= lo) & (values < hi))),
             "routed": int(np.sum(selected & (values >= lo) & (values < hi))),
             "routed_clean": int(np.sum(clean & (values >= lo) & (values < hi)))}
            for lo, hi in bins]
    result["strata"]["used_band_error"] = {}
    for tier, level in LEVEL.items():
        error = used_error(table, tier)
        take = mask & (routed["tier"] == level)
        result["strata"]["used_band_error"][tier] = [
            {"bin": [lo, hi if np.isfinite(hi) else None],
             "selected": int(np.sum(take & (error >= lo) & (error < hi)))}
            for lo, hi in ERROR_BINS]
    return result


def field_result(config_path, route):
    task = json.loads(Path(config_path).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    table = fitsio.read(str(path), columns=COLS)
    table = {name: table[name] for name in table.dtype.names}
    unique, dedup = unique_spectra(table)
    base = np.asarray(table["grz_base"], bool)
    labels = class_masks(table, unique & base)
    routed = route_rows(table, route)
    feh_cal, valid_cal = rvs_metallicity(table["desi_feh"], table["desi_teff"],
                                         table["desi_logg"])
    valid_abundance = (labels["cool_dwarf"] & valid_cal &
                       np.isfinite(table["desi_feh_err"]) &
                       (table["desi_feh_err"] > 0))
    poor = valid_abundance & (feh_cal < -1.5)
    result = {"field": task["name"], "path": str(path),
              "deduplication": dedup,
              "selection_scope": "One Legacy counterpart per unambiguous DESI object ID; labels conditional on DESI target selection and grz_base, not candidate purity",
              "label_thresholds": "DESI quality, stellar classification, finite Teff/logg, 3000<=Teff<5000 K and logg>=3.5 dwarf/<3.5 giant; nonstars from unambiguous known QSO/galaxy flag",
              "class_conditional_not_purity": True,
              "labels": {name: class_result(mask, table, routed)
                         for name, mask in labels.items()},
              "metallicity": {"calibration_domain": "4200<=Teff<6600 K, 1.8<=logg<5.5, finite Fe/H and positive finite Fe/H error",
                               "cool_dwarf_valid_domain": int(valid_abundance.sum()),
                               "calibrated_FeH_below_minus_1p5": class_result(poor, table, routed),
                               "raw_FeH_below_minus_1p5_valid_domain": int(np.sum(
                                   valid_abundance & (table["desi_feh"] < -1.5)))}}
    print(json.dumps({"field": task["name"], "labels": {name: value["n_labeled_in_grz_base"]
                        for name, value in result["labels"].items()},
                      "metal_poor": int(poor.sum())}), flush=True)
    return result


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    registration()
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Published cold-v4 preparation changed")
    prepared = json.loads(PREPARED.read_text())
    path = OUT / "route.joblib"
    if sha(path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen route changed")
    route = joblib.load(path)
    summary = {"prepared_sha256": sha(PREPARED), "fields": {},
               "compute_host": socket.gethostname()}
    for config in (CAL_CONFIG, *FINAL_CONFIGS):
        result = field_result(config, route)
        summary["fields"][result["field"]] = result
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a bounded DESI–Gaia–Legacy same-star sample without fitting a model."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.audit import checked_gaia_color, rvs_metallicity
from photdis.class_evidence import desi_class_evidence
from photdis.stage1 import require_compute, write_json
from cold_v4 import sha

CONFIG = Path("configs/joint_calibration_fields.json")
OUTPUT = Path("data/joint_calibration_v1/same_star.fits")
MANIFEST = Path("manifests/joint_calibration_v1.json")
COLUMNS = ["ls_id", "source_id", "ra", "dec", "ebv", "g0", "r", "z0",
           "gr0", "rz0", "sigma_g", "sigma_r", "sigma_z", "grz_base",
           "grz_clean", "desi_match", "desi_match_ambiguous", "desi_object_id",
           "desi_snr", "desi_separation_arcsec", "desi_quality",
           "desi_quality_reason", "desi_is_stellar", "desi_known_contaminant",
           "desi_teff", "desi_teff_err", "desi_logg", "desi_logg_err",
           "desi_feh", "desi_feh_err", "gaia_match", "gaia_parallax",
           "gaia_parallax_error", "gaia_phot_g_mean_mag", "gaia_bp_rp",
           "gaia_phot_bp_mean_flux", "gaia_phot_bp_mean_flux_error",
           "gaia_phot_bp_mean_mag", "gaia_phot_bp_n_obs",
           "gaia_phot_rp_mean_flux", "gaia_phot_rp_mean_flux_error",
           "gaia_phot_rp_mean_mag", "gaia_phot_rp_n_obs",
           "astro_quality", "zpt_valid", "p_corr", "snr_corr"]


def _count(mask):
    return int(np.sum(mask))


def _bins(mask, values, edges):
    finite = np.isfinite(values)
    return [{"range": [lo, hi], "count": _count(mask & finite &
            (values >= lo) & (values < hi))} for lo, hi in zip(edges[:-1], edges[1:])]


def select_unique(rows):
    """One unambiguous Legacy counterpart per DESI object and Gaia source."""
    ids = np.char.strip(rows["desi_object_id"].astype("U"))
    good = (rows["desi_match"] & ~rows["desi_match_ambiguous"] & (ids != ""))
    index = np.flatnonzero(good)
    snr = np.nan_to_num(rows["desi_snr"][index].astype(float), nan=-np.inf)
    sep = np.nan_to_num(rows["desi_separation_arcsec"][index].astype(float), nan=np.inf)
    order = np.lexsort((rows["ls_id"][index], sep, -snr, ids[index]))
    ranked = index[order]
    _, first = np.unique(ids[ranked], return_index=True)
    distinct = ranked[first]
    # A source may have multiple qualified DESI observations; keep the best.
    sid = rows["source_id"][distinct]
    # Missing Gaia identity must not merge unrelated DESI targets.
    key = np.where(sid > 0, sid, -rows["ls_id"][distinct])
    snr2 = np.nan_to_num(rows["desi_snr"][distinct].astype(float), nan=-np.inf)
    order2 = np.lexsort((rows["ls_id"][distinct], -snr2, key))
    ranked2 = distinct[order2]
    _, first2 = np.unique(key[order2], return_index=True)
    return np.sort(ranked2[first2]), {
        "matched_unambiguous_rows": len(index),
        "unique_desi_objects": len(distinct), "unique_source_rows": len(first2)}


def derive(rows):
    n = len(rows)
    evidence = desi_class_evidence({name: rows[name] for name in rows.dtype.names})
    gaia = {name[5:]: rows[name] for name in rows.dtype.names
            if name.startswith("gaia_")}
    gaia["gaia_match"] = rows["gaia_match"]
    color = checked_gaia_color(gaia)
    feh_cal, feh_domain = rvs_metallicity(rows["desi_feh"], rows["desi_teff"],
                                          rows["desi_logg"])
    feh_valid = (feh_domain & np.isfinite(rows["desi_feh_err"])
                 & (rows["desi_feh_err"] > 0))
    feh_cal[~feh_valid] = np.nan
    photo = (rows["grz_base"] & rows["grz_clean"] & np.isfinite(rows["ebv"])
             & (rows["ebv"] < .03))
    finite_parallax = (rows["gaia_match"] & np.isfinite(rows["gaia_parallax"])
                       & np.isfinite(rows["gaia_parallax_error"])
                       & (rows["gaia_parallax_error"] > 0))
    astro = finite_parallax & rows["astro_quality"] & rows["zpt_valid"]
    high = astro & photo & (rows["snr_corr"] >= 20)
    mid = astro & photo & (rows["snr_corr"] >= 10) & (rows["snr_corr"] < 20)
    weak = astro & photo & (rows["snr_corr"] >= 0) & (rows["snr_corr"] < 10)
    negative = astro & photo & (rows["snr_corr"] < 0)
    mg = np.full(n, np.nan, np.float32)
    sigma_mg = np.full(n, np.nan, np.float32)
    mg[high] = rows["g0"][high] - 10 + 5 * np.log10(rows["p_corr"][high])
    sigma_mg[high] = np.hypot(rows["sigma_g"][high],
                              5 / np.log(10) / rows["snr_corr"][high])
    gaia_mg = np.full(n, np.nan, np.float32)
    cmd_ok = astro & (rows["snr_corr"] >= 10) & color["bp_rp_valid"]
    gaia_mg[cmd_ok] = rows["gaia_phot_g_mean_mag"][cmd_ok] - 10 + 5 * np.log10(
        rows["p_corr"][cmd_ok])
    wd_cmd = (cmd_ok & (color["bp_rp_checked"] < 1.5)
              & (gaia_mg > 10 + 2 * color["bp_rp_checked"]))
    label = evidence["class_evidence"].copy()
    label[wd_cmd & (label == b"other_star_or_transition")] = b"wd_cmd_candidate"
    dtype = [("field", "S48"), ("role", "S24"),
             *[(name, rows.dtype[name]) for name in rows.dtype.names],
             ("class_evidence", "S28"), ("class_evidence_source", "S12"),
             ("class_evidence_quality", "S28"),
             ("class_conflicts_cold_dwarf", "?"),
             ("bp_rp_checked", "f4"), ("wd_cmd_candidate", "?"),
             ("feh_cal", "f4"), ("feh_cal_valid", "?"),
             ("clean_low_dust", "?"), ("astrometry_usable", "?"),
             ("parallax_layer", "S16"), ("mg_high_snr", "f4"),
             ("sigma_mg_high_snr", "f4")]
    out = np.empty(n, dtype=dtype)
    for name in rows.dtype.names:
        out[name] = rows[name]
    out["class_evidence"] = label
    for name in ("class_evidence_source", "class_evidence_quality",
                 "class_conflicts_cold_dwarf"):
        out[name] = evidence[name]
    out["bp_rp_checked"] = color["bp_rp_checked"]
    out["wd_cmd_candidate"] = wd_cmd
    out["feh_cal"] = feh_cal
    out["feh_cal_valid"] = feh_valid
    out["clean_low_dust"] = photo
    out["astrometry_usable"] = astro
    out["parallax_layer"] = b"unusable"
    out["parallax_layer"][negative] = b"negative"
    out["parallax_layer"][weak] = b"weak_0_10"
    out["parallax_layer"][mid] = b"mid_10_20"
    out["parallax_layer"][high] = b"high_ge20"
    out["mg_high_snr"] = mg
    out["sigma_mg_high_snr"] = sigma_mg
    return out


def summarize(rows):
    labels = np.unique(rows["class_evidence"])
    result = {"rows": len(rows), "class_counts": {}, "parallax_layers": {},
              "high_snr_core_by_feh": {}, "core_coverage": {}}
    for name in labels:
        text = name.decode()
        class_mask = rows["class_evidence"] == name
        result["class_counts"][text] = {"all": _count(class_mask),
            "clean_low_dust": _count(class_mask & rows["clean_low_dust"]),
            "high_snr": _count(class_mask & (rows["parallax_layer"] == b"high_ge20")),
            "calibrated_feh": _count(class_mask & rows["feh_cal_valid"])}
    for name in (b"high_ge20", b"mid_10_20", b"weak_0_10", b"negative", b"unusable"):
        result["parallax_layers"][name.decode()] = _count(rows["parallax_layer"] == name)
    core = rows["class_evidence"] == b"cool_main_sequence"
    core_high = core & (rows["parallax_layer"] == b"high_ge20") & rows["feh_cal_valid"]
    result["high_snr_core_by_feh"] = _bins(core_high, rows["feh_cal"],
                                               [-5, -2, -1.5, -1, -.5, 0, 1])
    for name, values, edges in (("gr0", rows["gr0"], [0, .6, .909, 1.2, 1.6, 3]),
                                ("r", rows["r"], [12, 14, 16, 18, 20, 22, 26]),
                                ("teff", rows["desi_teff"], [4200, 4400, 4600, 4800, 5000])):
        result["core_coverage"][name] = {
            "all": _bins(core & rows["clean_low_dust"], values, edges),
            "high_snr_with_feh": _bins(core_high, values, edges)}
    return result


def main():
    require_compute()
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("Joint inventory v1 already exists")
    config = json.loads(CONFIG.read_text())
    pieces = []
    field_counts = {}
    for entry in config["fields"]:
        field = json.loads(Path(entry["config"]).read_text())
        path = Path(field["output_dir"]) / "field_diagnostics.fits"
        selected = []
        with fitsio.FITS(str(path)) as file:
            nrows = file[1].get_nrows()
            for start in range(0, nrows, 100000):
                raw = file[1].read(columns=COLUMNS,
                    rows=np.arange(start, min(start + 100000, nrows)))
                selected.append(raw[raw["desi_match"] & ~raw["desi_match_ambiguous"]])
        rows = np.concatenate(selected)
        idx, dedup = select_unique(rows)
        out = derive(rows[idx])
        out["field"] = field["name"].encode()
        out["role"] = entry["role"].encode()
        pieces.append(out)
        field_counts[field["name"]] = {"role": entry["role"], "input_rows": nrows,
            "desi_matched_rows": len(rows), "deduplication": dedup,
            "summary": summarize(out)}
        print(json.dumps({"field": field["name"], "desi_unique": len(out)}), flush=True)
    all_rows = np.concatenate(pieces)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(OUTPUT), all_rows, extname="JOINT", clobber=False)
    manifest = {"scope": config["purpose"], "config_sha256": sha(CONFIG),
                "output": str(OUTPUT), "output_sha256": sha(OUTPUT),
                "output_rows": len(all_rows), "fields": field_counts,
                "by_role": {role: summarize(all_rows[all_rows["role"] == role.encode()])
                            for role in ("development", "reserved_validation")},
                "all": summarize(all_rows), "compute_host": socket.gethostname()}
    write_json(MANIFEST, manifest)
    print(json.dumps({"output_rows": len(all_rows), "summary": manifest["all"]}), flush=True)


if __name__ == "__main__":
    main()

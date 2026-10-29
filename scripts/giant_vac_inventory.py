#!/usr/bin/env python3
"""Join qualified DESI giants to the two published distance VACs by identity."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.holdouts import load_reference_catalogs, annotate_holdouts
from photdis.stage1 import require_compute, write_json

JOINT = [Path("data/joint_calibration_v1_evidence_v2/same_star.fits"),
         Path("data/joint_calibration_targeted_v1_evidence_v2/same_star.fits")]
SPEC = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-specdis/v2.1/iron-yr1-v2.1.fits")
SP = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-spdist/v1.0/mws-iron-SPDist-240613.fits")
OUT = Path("data/giant_vac_v3/same_star.fits")
MANIFEST = Path("manifests/giant_vac_v3.json")


def exact_pair_join(query_target, query_source, catalog):
    """Map (DESI target, Gaia source) to a unique row; ambiguous pairs stay unmatched."""
    order = np.argsort(catalog["TARGETID"], kind="stable")
    target = catalog["TARGETID"][order]
    lower = np.searchsorted(target, query_target, side="left")
    upper = np.searchsorted(target, query_target, side="right")
    result = np.full(len(query_target), -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if query_source[i] <= 0 or lo == hi:
            continue
        matches = order[lo:hi][catalog["SOURCE_ID"][order[lo:hi]] == query_source[i]]
        if len(matches) == 1:
            result[i] = matches[0]
    return result


def giant_mask(rows):
    return np.char.strip(rows["class_evidence"].astype("U")) == "conservative_giant"


def deduplicate_giant_identities(rows):
    """Choose one row per Gaia star and keep both join keys row-aligned."""
    source = rows["source_id"]
    target = np.char.strip(rows["desi_object_id"].astype("U")).astype(np.int64)
    identity = np.where(source > 0, source, -target)
    order = np.lexsort((rows["ls_id"],
                        -np.nan_to_num(rows["desi_snr"], nan=-1), identity))
    _, first = np.unique(identity[order], return_index=True)
    selected = rows[order[first]]
    return selected, np.char.strip(selected["desi_object_id"].astype("U")).astype(np.int64), selected["source_id"]


def summarize(rows):
    giant = giant_mask(rows)
    clean = giant & rows["clean_low_dust"] & ~rows["cmd_spectral_conflict"]
    ranges = {"lt4200": rows["desi_teff"] < 4200,
              "4200_5000": (rows["desi_teff"] >= 4200) & (rows["desi_teff"] < 5000),
              "5000_7000": (rows["desi_teff"] >= 5000) & (rows["desi_teff"] < 7000),
              "logg_lt1p8": rows["desi_logg"] < 1.8,
              "feh_cal_valid": rows["abundance_calibration_valid"]}
    result = {"all": int(len(rows)), "qualified_giant": int(giant.sum()),
              "clean_giant": int(clean.sum()),
              "specdis_match": int((clean & rows["specdis_match"]).sum()),
              "spdist_match": int((clean & rows["spdist_match"]).sum()),
              "specdis_valid": int((clean & rows["specdis_valid"]).sum()),
              "spdist_valid": int((clean & rows["spdist_valid"]).sum()),
              "both_valid": int((clean & rows["specdis_valid"] & rows["spdist_valid"]).sum()),
              "giant_cluster_members": int((clean & (rows["cluster_group"] > 0)).sum()),
              "by_type": {name: {"clean": int((clean & mask).sum()),
                                  "specdis": int((clean & mask & rows["specdis_valid"]).sum()),
                                  "spdist": int((clean & mask & rows["spdist_valid"]).sum())}
                          for name, mask in ranges.items()}}
    groups, count = np.unique(rows["cluster_group"][clean & (rows["cluster_group"] > 0)],
                              return_counts=True)
    result["cluster_groups"] = {str(int(group)): int(n) for group, n in zip(groups, count)}
    return result


def main():
    require_compute()
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError("Giant VAC inventory already exists")
    arrays = [fitsio.read(str(path)) for path in JOINT]
    all_rows = np.concatenate(arrays)
    giant = giant_mask(all_rows)
    rows, target, sid = deduplicate_giant_identities(all_rows[giant])

    spec = fitsio.read(str(SPEC), ext="SPECDIS", columns=[
        "TARGETID", "SOURCE_ID", "DIST", "DISTERR", "BINARY_FLAG", "BINARY_POSSIBILITY"])
    sp = fitsio.read(str(SP), ext="SPDIST", columns=[
        "TARGETID", "SOURCE_ID", "FLAG_GOOD", "QFLAG_INPUT", "G0", "E_GMAG",
        "MG_16", "MG_50", "MG_84", "TEFF_COR", "LOGG_COR"])
    ispec = exact_pair_join(target, sid, spec)
    isp = exact_pair_join(target, sid, sp)
    for table, index in ((spec, ispec), (sp, isp)):
        present = index >= 0
        if not (np.array_equal(table["TARGETID"][index[present]], target[present])
                and np.array_equal(table["SOURCE_ID"][index[present]], sid[present])):
            raise AssertionError("Distance VAC identity join is inconsistent")
    dtype = rows.dtype.descr + [
        ("specdis_match", "?"), ("specdis_valid", "?"),
        ("specdis_dist_kpc", "f4"), ("specdis_disterr_kpc", "f4"),
        ("specdis_binary_flag", "i2"), ("specdis_mg_legacy", "f4"),
        ("specdis_sigma_mg", "f4"),
        ("spdist_match", "?"), ("spdist_valid", "?"),
        ("spdist_flag_good", "?"), ("spdist_qflag_input", "?"),
        ("spdist_mg50_gaia", "f4"), ("spdist_g0_gaia", "f4"),
        ("spdist_mg_legacy", "f4"), ("spdist_sigma_mg", "f4"),
        ("spdist_dist_kpc", "f4"), ("cluster_group", "i8")]
    out = np.empty(len(rows), dtype=dtype)
    for name in rows.dtype.names:
        out[name] = rows[name]
    for name in ("specdis_dist_kpc", "specdis_disterr_kpc", "specdis_mg_legacy",
                 "specdis_sigma_mg", "spdist_mg50_gaia", "spdist_g0_gaia",
                 "spdist_mg_legacy", "spdist_sigma_mg", "spdist_dist_kpc"):
        out[name] = np.nan
    out["specdis_binary_flag"] = -1
    out["specdis_match"] = ispec >= 0
    out["spdist_match"] = isp >= 0
    out["specdis_valid"] = False
    out["spdist_valid"] = False
    out["spdist_flag_good"] = False
    out["spdist_qflag_input"] = False
    m = ispec >= 0
    out["specdis_dist_kpc"][m] = spec["DIST"][ispec[m]]
    out["specdis_disterr_kpc"][m] = spec["DISTERR"][ispec[m]]
    out["specdis_binary_flag"][m] = spec["BINARY_FLAG"][ispec[m]]
    d, de = out["specdis_dist_kpc"].astype(float), out["specdis_disterr_kpc"].astype(float)
    valid = m & np.isfinite(d) & np.isfinite(de) & (d > 0) & (de > 0) & (de < d)
    valid &= np.isfinite(rows["g0"]) & np.isfinite(rows["sigma_g"])
    valid &= out["specdis_binary_flag"] == 1
    out["specdis_valid"] = valid
    out["specdis_mg_legacy"][valid] = rows["g0"][valid] - 10 - 5 * np.log10(d[valid])
    out["specdis_sigma_mg"][valid] = np.hypot(rows["sigma_g"][valid],
                                                 5 / np.log(10) * de[valid] / d[valid])
    m = isp >= 0
    out["spdist_flag_good"][m] = sp["FLAG_GOOD"][isp[m]]
    out["spdist_qflag_input"][m] = sp["QFLAG_INPUT"][isp[m]]
    out["spdist_mg50_gaia"][m] = sp["MG_50"][isp[m]]
    out["spdist_g0_gaia"][m] = sp["G0"][isp[m]]
    g0, mg50 = out["spdist_g0_gaia"].astype(float), out["spdist_mg50_gaia"].astype(float)
    mag16 = np.full(len(rows), np.nan)
    mag84 = np.full(len(rows), np.nan)
    em = np.full(len(rows), np.nan)
    mag16[m], mag84[m], em[m] = sp["MG_16"][isp[m]], sp["MG_84"][isp[m]], sp["E_GMAG"][isp[m]]
    valid = (m & out["spdist_flag_good"] & out["spdist_qflag_input"]
             & np.isfinite(g0) & np.isfinite(mg50) & np.isfinite(mag16)
             & np.isfinite(mag84) & np.isfinite(em) & (em >= 0)
             & (10 < g0) & (g0 < 30) & (-8 < mg50) & (mg50 < 15)
             & (mag84 >= mag16) & (mag84 - mag16 < 4)
             & np.isfinite(rows["g0"]) & np.isfinite(rows["sigma_g"]))
    out["spdist_valid"] = valid
    mu = g0[valid] - mg50[valid]
    out["spdist_mg_legacy"][valid] = rows["g0"][valid] - mu
    out["spdist_dist_kpc"][valid] = 10 ** ((mu - 10) / 5)
    out["spdist_sigma_mg"][valid] = np.sqrt(
        rows["sigma_g"][valid] ** 2 + em[valid] ** 2
        + ((mag84[valid] - mag16[valid]) / 2) ** 2 + 0.167 ** 2)
    refs = load_reference_catalogs("data/reference")
    hold = annotate_holdouts(refs, rows["source_id"], rows["ra"], rows["dec"])
    out["cluster_group"] = hold["cluster_member_group"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(OUT), out, extname="GIANTVAC", clobber=False)
    summary = summarize(out)
    manifest = {"input": [str(path) for path in JOINT],
                "specdis": str(SPEC), "spdist": str(SP),
                "join_algorithm": "TARGETID range search, unique exact (TARGETID,SOURCE_ID) match; Gaia SOURCE_ID>0",
                "output": str(OUT), "summary": summary,
                "compute_host": socket.gethostname()}
    write_json(MANIFEST, manifest)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Whole-system NGC 2419 cool-giant check with actual DR11 grz measurements."""
from __future__ import annotations

import io
import json
from pathlib import Path
import socket
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from giant_vac_inventory import SPEC, SP, exact_pair_join
from photdis.photometry import derive_photometry
from photdis.stage1 import (CORE_COLUMNS, PHOT_COLUMNS, legacy_ids,
                           require_compute, write_json)

MEMBERS = Path("data/reference/vasiliev_baumgardt_2021_members.zip")
SWEEPS = Path("/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/north/sweep/11.0")
MWS = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws/iron/v1.0/mwsall-pix-iron.fits")
OUT = Path("manifests/giant_ngc2419_reference_v1.json")
FIELD = (110, 35, 120, 45)
SWEEP_FILES = tuple(
    f"sweep-{ra:03d}p{dec:03d}-{ra+5:03d}p{dec+5:03d}.fits"
    for ra in (110, 115) for dec in (35, 40))
REFERENCE = {"mu_mag": 19.71, "mu_sigma_mag": .08,
             "method": "RR Lyrae variables; Di Criscienzo et al. 2011",
             "url": "https://arxiv.org/abs/1011.5398"}


def member_ids():
    with zipfile.ZipFile(MEMBERS) as archive:
        with archive.open("catalogues/NGC_2419.txt") as stream:
            return np.unique(np.asarray([
                int(parts[0]) for line in io.TextIOWrapper(stream)
                if (parts := line.split()) and not line.startswith("#")
                and float(parts[-1]) >= .5], np.int64))


def rvs_giants(ids):
    rvs = fitsio.read(str(MWS), ext="RVTAB", columns=[
        "TARGETID", "PRIMARY", "RR_SPECTYPE", "RVS_WARN", "SN_R", "VSINI",
        "TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "SUCCESS"])
    gaia = fitsio.read(str(MWS), ext="GAIA", columns=["SOURCE_ID"])
    if len(rvs) != len(gaia):
        raise ValueError("MWS RVS and GAIA extensions are not row aligned")
    present = np.isin(gaia["SOURCE_ID"], ids)
    good = (present & rvs["PRIMARY"]
            & (np.char.strip(rvs["RR_SPECTYPE"].astype("U")) == "STAR")
            & (rvs["RVS_WARN"] == 0) & rvs["SUCCESS"]
            & np.isfinite(rvs["SN_R"]) & (rvs["SN_R"] >= 10)
            & np.isfinite(rvs["VSINI"]) & (rvs["VSINI"] >= 0)
            & (rvs["VSINI"] < 30)
            & np.isfinite(rvs["TEFF"]) & (rvs["TEFF"] > 0)
            & (rvs["TEFF"] < 7000) & np.isfinite(rvs["TEFF_ERR"])
            & (rvs["TEFF_ERR"] > 0) & np.isfinite(rvs["LOGG"])
            & np.isfinite(rvs["LOGG_ERR"]) & (rvs["LOGG_ERR"] > 0)
            & (rvs["LOGG"] + rvs["LOGG_ERR"] < 3.2))
    idx = np.flatnonzero(good)
    if len(idx):
        order = np.lexsort((-rvs["SN_R"][idx], gaia["SOURCE_ID"][idx]))
        idx = idx[order]
        _, first = np.unique(gaia["SOURCE_ID"][idx], return_index=True)
        idx = idx[first]
    return rvs[idx], gaia["SOURCE_ID"][idx]


def dr11_members(ids):
    parts = []
    for filename in SWEEP_FILES:
        path = SWEEPS / filename
        small = fitsio.read(str(path), ext=1, columns=["REF_ID", "REF_CAT", "RA", "DEC"])
        inside = ((small["RA"] >= FIELD[0]) & (small["RA"] < FIELD[2])
                  & (small["DEC"] >= FIELD[1]) & (small["DEC"] < FIELD[3]))
        row_index = np.flatnonzero(
            inside & (np.char.strip(small["REF_CAT"].astype("U")) == "G3")
            & np.isin(small["REF_ID"], ids))
        if len(row_index):
            parts.append(fitsio.read(str(path), ext=1,
                                     columns=CORE_COLUMNS + PHOT_COLUMNS,
                                     rows=row_index))
    if not parts:
        raise ValueError("NGC 2419 has no Gaia-linked DR11 member rows")
    rows = np.concatenate(parts)
    photo = derive_photometry(rows)
    sid = rows["REF_ID"].astype(np.int64)
    # Prefer the cleanest one of any repeated Legacy measurements of a Gaia star.
    score = (100 * photo["grz_clean"] + 10 * photo["grz_base"]
             + np.nan_to_num(photo["snr_g"], nan=0, posinf=0) / 1000)
    order = np.lexsort((-score, sid))
    _, first = np.unique(sid[order], return_index=True)
    selected = order[first]
    return rows[selected], {key: val[selected] for key, val in photo.items()}, len(sid)


def valid_catalogs(source_ids, target_ids):
    spec = fitsio.read(str(SPEC), ext="SPECDIS", columns=[
        "TARGETID", "SOURCE_ID", "DIST", "DISTERR", "BINARY_FLAG"])
    sp = fitsio.read(str(SP), ext="SPDIST", columns=[
        "TARGETID", "SOURCE_ID", "FLAG_GOOD", "QFLAG_INPUT",
        "G0", "MG_16", "MG_50", "MG_84"])
    return (spec, exact_pair_join(target_ids, source_ids, spec),
            sp, exact_pair_join(target_ids, source_ids, sp))


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    ids = member_ids()
    rvs, giant_ids = rvs_giants(ids)
    rows, photo, raw_count = dr11_members(ids)
    sid = rows["REF_ID"].astype(np.int64)
    lookup = {int(source): i for i, source in enumerate(sid)}
    linked = [i for i, source in enumerate(giant_ids) if int(source) in lookup]
    gsid = giant_ids[linked]
    target_ids = rvs["TARGETID"][linked]
    spec, ispec, sp, isp = valid_catalogs(gsid, target_ids)
    stars = []
    for k, j in enumerate(linked):
        p = lookup[int(giant_ids[j])]
        entry = {"source_id": int(gsid[k]), "targetid": int(target_ids[k]),
                 "ls_id": int(legacy_ids(rows[p:p+1])[0]),
                 "teff": float(rvs["TEFF"][j]), "logg": float(rvs["LOGG"][j]),
                 "desi_snr": float(rvs["SN_R"][j]), "ebv": float(rows["EBV"][p]),
                 "grz_base": bool(photo["grz_base"][p]),
                 "grz_clean": bool(photo["grz_clean"][p]),
                 "g0": float(photo["g0"][p]),
                 "gr0": float(photo["gr0"][p]),
                 "rz0": float(photo["rz0"][p]),
                 "sigma_g": float(photo["sigma_g"][p]),
                 "mu_rr": REFERENCE["mu_mag"]}
        if ispec[k] >= 0:
            v = spec[ispec[k]]
            d, de = float(v["DIST"]), float(v["DISTERR"])
            if (int(v["BINARY_FLAG"]) == 1 and np.isfinite(d)
                    and np.isfinite(de) and d > 0 and 0 < de < d):
                entry["specdis_mu"] = float(10 + 5 * np.log10(d))
                entry["specdis_sigma_mu"] = float(5 / np.log(10) * de / d)
        if isp[k] >= 0:
            v = sp[isp[k]]
            if (v["FLAG_GOOD"] and v["QFLAG_INPUT"]
                    and np.isfinite(v["G0"]) and np.isfinite(v["MG_50"])):
                entry["spdist_mu"] = float(v["G0"] - v["MG_50"])
                entry["spdist_sigma_mu"] = float(np.hypot(
                    (v["MG_84"] - v["MG_16"]) / 2, .167))
        stars.append(entry)
    clean = [star for star in stars if star["grz_clean"]]
    result = {"system": "NGC_2419", "field_bounds_deg": FIELD,
              "sweeps": list(SWEEP_FILES), "reference": REFERENCE,
              "member_source": str(MEMBERS), "membership_probability_min": .5,
              "counts": {"published_probable_members": len(ids),
                         "dr11_member_rows": raw_count,
                         "dr11_member_unique": len(sid),
                         "qualified_rvs_giant": len(giant_ids),
                         "giant_with_dr11": len(stars),
                         "giant_clean_grz": len(clean),
                         "cool_lt5200_clean": sum(s["teff"] < 5200 for s in clean),
                         "clean_specdis_valid": sum("specdis_mu" in s for s in clean),
                         "clean_spdist_valid": sum("spdist_mu" in s for s in clean)},
              "stars": stars, "compute_host": socket.gethostname()}
    write_json(OUT, result)
    print(json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()

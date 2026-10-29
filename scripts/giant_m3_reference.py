#!/usr/bin/env python3
"""Check independent M3 RR Lyrae distance against DESI giant and VAC overlap."""
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

from photdis.stage1 import require_compute, write_json
from audit_giant_vac_identity import match_one

FIELD = Path("data/field_sky_ra200_210_v1/field_diagnostics.fits")
MEMBERS = Path("data/reference/vasiliev_baumgardt_2021_members.zip")
SPEC = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-specdis/v2.1/iron-yr1-v2.1.fits")
SP = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-spdist/v1.0/mws-iron-SPDist-240613.fits")
OUT = Path("manifests/giant_m3_reference_v1.json")
SYSTEM = "NGC_5272_M_3"
MU_RR = 15.041
SIGMA_STAT = 0.017
SIGMA_SYS = 0.036
REFERENCE = "https://arxiv.org/abs/2008.11745"


def m3_member_ids(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"catalogues/{SYSTEM}.txt") as stream:
            members = [int(parts[0]) for line in io.TextIOWrapper(stream)
                       if (parts := line.split()) and not line.startswith("#")
                       and float(parts[-1]) >= 0.5]
    return np.unique(np.asarray(members, np.int64))


def summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    center = np.median(values)
    return {"n": len(values), "median": float(center),
            "nmad": float(1.4826 * np.median(np.abs(values - center))),
            "p16": float(np.percentile(values, 16)),
            "p84": float(np.percentile(values, 84))}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    ids = m3_member_ids(MEMBERS)
    cols = ["ls_id", "source_id", "ra", "dec", "ebv", "g0", "r0", "z0",
            "sigma_g", "grz_base", "grz_clean", "desi_match",
            "desi_match_ambiguous", "desi_quality", "desi_snr", "desi_teff",
            "desi_logg", "desi_logg_err"]
    field = fitsio.read(str(FIELD), columns=cols)
    matched = field[np.isin(field["source_id"], ids)]
    if len(matched):
        quality = (matched["grz_clean"].astype(int) * 10
                   - np.nan_to_num(matched["sigma_g"], nan=10))
        order = np.lexsort((-quality, matched["source_id"]))
        _, first = np.unique(matched["source_id"][order], return_index=True)
        matched = matched[order[first]]
    giant = (matched["desi_match"] & ~matched["desi_match_ambiguous"]
             & matched["desi_quality"] & np.isfinite(matched["desi_teff"])
             & (matched["desi_teff"] > 0) & (matched["desi_teff"] < 7000)
             & np.isfinite(matched["desi_logg"])
             & np.isfinite(matched["desi_logg_err"])
             & (matched["desi_logg_err"] > 0)
             & (matched["desi_logg"] + matched["desi_logg_err"] < 3.2))
    clean = giant & matched["grz_clean"] & np.isfinite(matched["g0"])
    clean &= (matched["ebv"] >= 0) & (matched["ebv"] < 0.03)
    target = matched[clean]
    rows = []
    for row in target:
        rows.append({"source_id": int(row["source_id"]), "ls_id": int(row["ls_id"]),
                     "ra": float(row["ra"]), "dec": float(row["dec"]),
                     "g0": float(row["g0"]), "gr0": float(row["g0"]-row["r0"]),
                     "rz0": float(row["r0"]-row["z0"]),
                     "mg_legacy_rr": float(row["g0"]-MU_RR),
                     "desi_teff": float(row["desi_teff"]),
                     "desi_logg": float(row["desi_logg"]),
                     "desi_snr": float(row["desi_snr"])})
    sid = target["source_id"]
    vac = {}
    for name, path, ext, columns in (
        ("specdis", SPEC, "SPECDIS", ["SOURCE_ID", "TARGETID", "DIST", "DISTERR", "BINARY_FLAG"]),
        ("spdist", SP, "SPDIST", ["SOURCE_ID", "TARGETID", "FLAG_GOOD", "QFLAG_INPUT",
                                   "G0", "MG_16", "MG_50", "MG_84"])):
        table = fitsio.read(str(path), ext=ext, columns=columns)
        index, unique = match_one(sid, table["SOURCE_ID"])
        found = (index >= 0) & unique
        delta = []
        if name == "specdis":
            valid = (found & (table["BINARY_FLAG"][np.maximum(index, 0)] == 1)
                     & np.isfinite(table["DIST"][np.maximum(index, 0)])
                     & (table["DIST"][np.maximum(index, 0)] > 0)
                     & np.isfinite(table["DISTERR"][np.maximum(index, 0)])
                     & (table["DISTERR"][np.maximum(index, 0)] > 0))
            for k in np.flatnonzero(valid):
                j = index[k]
                mu = 10 + 5 * np.log10(float(table["DIST"][j]))
                delta.append(mu - MU_RR)
                rows[k]["specdis_mu"] = float(mu)
                rows[k]["specdis_mu_error"] = float(5 / np.log(10) * table["DISTERR"][j] / table["DIST"][j])
        else:
            valid = (found & table["FLAG_GOOD"][np.maximum(index, 0)]
                     & table["QFLAG_INPUT"][np.maximum(index, 0)]
                     & np.isfinite(table["G0"][np.maximum(index, 0)])
                     & np.isfinite(table["MG_50"][np.maximum(index, 0)]))
            for k in np.flatnonzero(valid):
                j = index[k]
                mu = float(table["G0"][j] - table["MG_50"][j])
                delta.append(mu - MU_RR)
                rows[k]["spdist_mu"] = mu
                rows[k]["spdist_mu_error"] = float(np.hypot(
                    (table["MG_84"][j] - table["MG_16"][j]) / 2, .167))
        vac[name] = {"unique_source_match": int(found.sum()),
                     "quality_valid": int(valid.sum()),
                     "mu_minus_rr": summary(delta)}
    output = {"system": SYSTEM, "source": str(FIELD),
              "member_source": str(MEMBERS),
              "reference": {"url": REFERENCE, "mu_mag": MU_RR,
                            "statistical_sigma_mag": SIGMA_STAT,
                            "systematic_sigma_mag": SIGMA_SYS,
                            "calibration": "theoretical near-infrared RR Lyrae period-luminosity-metallicity"},
              "counts": {"published_probable_members": len(ids),
                         "field_member_unique": len(matched),
                         "grz_base": int(matched["grz_base"].sum()),
                         "grz_clean": int(matched["grz_clean"].sum()),
                         "qualified_desi_giant": int(giant.sum()),
                         "qualified_clean_low_dust_giant": len(target)},
              "vac": vac, "members": rows, "compute_host": socket.gethostname()}
    write_json(OUT, output)
    print(json.dumps({"counts": output["counts"], "vac": vac}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Find DESI-qualified giant members of published external cluster systems."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.holdouts import load_reference_catalogs
from photdis.stage1 import require_compute, write_json

MWS = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws/iron/v1.0/mwsall-pix-iron.fits")
REF = Path("data/reference")
OUT = Path("manifests/giant_cluster_system_ranking.json")


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    refs = load_reference_catalogs(REF)
    mws = fitsio.read(str(MWS), ext="RVTAB", columns=[
        "TARGETID", "PRIMARY", "RR_SPECTYPE", "RVS_WARN", "SN_R", "VSINI",
        "TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "SUCCESS"])
    gaia = fitsio.read(str(MWS), ext="GAIA", columns=["SOURCE_ID"])
    fiber = fitsio.read(str(MWS), ext="FIBERMAP", columns=["TARGETID", "TARGET_RA", "TARGET_DEC"])
    if len(mws) != len(gaia) or len(mws) != len(fiber):
        raise ValueError("MWS HDUs are not row aligned")
    if not np.array_equal(mws["TARGETID"], fiber["TARGETID"]):
        raise ValueError("MWS target IDs are not row aligned")
    pos = np.searchsorted(refs.member_source_id, gaia["SOURCE_ID"])
    safe = np.minimum(pos, len(refs.member_source_id) - 1)
    member = ((pos < len(refs.member_source_id))
              & (refs.member_source_id[safe] == gaia["SOURCE_ID"]))
    group = np.zeros(len(mws), np.int64)
    group[member] = refs.member_group[safe[member]]
    good = (member & mws["PRIMARY"] & (mws["RR_SPECTYPE"] == b"STAR")
            & (mws["RVS_WARN"] == 0) & mws["SUCCESS"]
            & np.isfinite(mws["SN_R"]) & (mws["SN_R"] >= 10)
            & np.isfinite(mws["VSINI"]) & (mws["VSINI"] >= 0)
            & (mws["VSINI"] < 30) & np.isfinite(mws["TEFF"])
            & (mws["TEFF"] > 0) & (mws["TEFF"] < 7000)
            & np.isfinite(mws["TEFF_ERR"]) & (mws["TEFF_ERR"] > 0)
            & np.isfinite(mws["LOGG"]) & np.isfinite(mws["LOGG_ERR"])
            & (mws["LOGG_ERR"] > 0)
            & (mws["LOGG"] + mws["LOGG_ERR"] < 3.2))
    names = json.loads((REF / "holdout_group_names.json").read_text())
    catalog = np.genfromtxt(REF / "vasiliev_baumgardt_2021_clusters.txt",
                            names=True, dtype=None, encoding="utf-8")
    result = []
    for group_id in np.unique(group[good]):
        if group_id < 100000:
            continue
        chosen = np.flatnonzero(good & (group == group_id))
        info = names[str(int(group_id))]
        gc = catalog[int(group_id) - 100000]
        source_ids = np.unique(gaia["SOURCE_ID"][chosen])
        result.append({"group": int(group_id), "name": info["name"],
                       "qualified_giant_spectra": len(chosen),
                       "unique_gaia_members": len(source_ids),
                       "cool_lt5200": int(np.sum(mws["TEFF"][chosen] < 5200)),
                       "ra": float(gc["ra"]), "dec": float(gc["dec"]),
                       "distance_kpc": float(gc["dist"]),
                       "distance_error_kpc": float(gc["disterr"]),
                       "member_source_ids": source_ids.astype(int).tolist()})
    result.sort(key=lambda item: (-item["cool_lt5200"], -item["unique_gaia_members"]))
    write_json(OUT, {"source": str(MWS), "reference": str(REF),
                     "qualified_giant_members": int(good.sum()),
                     "systems": result, "compute_host": socket.gethostname()})
    print(json.dumps({"systems": len(result), "top": result[:5]}), flush=True)


if __name__ == "__main__":
    main()

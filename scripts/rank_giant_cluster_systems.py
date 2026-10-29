#!/usr/bin/env python3
"""Find DESI-qualified giant members of published external cluster systems."""
from __future__ import annotations

import io
from pathlib import Path
import socket
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.holdouts import GC_ARCHIVE_ALIASES
from photdis.stage1 import require_compute, write_json

MWS = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws/iron/v1.0/mwsall-pix-iron.fits")
REF = Path("data/reference")
OUT = Path("manifests/giant_cluster_system_ranking_v2.json")


def memberships(catalog, archive_path):
    """Read globular memberships directly, including stars shared with Hunt.

    The normalized holdout table assigns each Gaia ID one group, so a Hunt
    group may take precedence over the Vasiliev group for the same cluster.
    """
    ids, groups = [], []
    with zipfile.ZipFile(archive_path) as archive:
        for i, row in enumerate(catalog):
            name = GC_ARCHIVE_ALIASES.get(str(row["Name"]), str(row["Name"]))
            with archive.open(f"catalogues/{name}.txt") as stream:
                for line in io.TextIOWrapper(stream):
                    if not line.strip() or line.startswith("#"):
                        continue
                    values = line.split()
                    if float(values[-1]) >= .5:
                        ids.append(int(values[0]))
                        groups.append(100000 + i)
    ids = np.asarray(ids, np.int64)
    groups = np.asarray(groups, np.int64)
    order = np.argsort(ids, kind="stable")
    ids, groups = ids[order], groups[order]
    unique = np.r_[True, ids[1:] != ids[:-1]] & np.r_[ids[:-1] != ids[1:], True]
    return ids[unique], groups[unique], int((~unique).sum())


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    catalog = np.genfromtxt(REF / "vasiliev_baumgardt_2021_clusters.txt",
                            names=True, dtype=None, encoding="utf-8")
    ids, member_groups, ambiguous = memberships(
        catalog, REF / "vasiliev_baumgardt_2021_members.zip")
    mws = fitsio.read(str(MWS), ext="RVTAB", columns=[
        "TARGETID", "PRIMARY", "RR_SPECTYPE", "RVS_WARN", "SN_R", "VSINI",
        "TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "SUCCESS"])
    gaia = fitsio.read(str(MWS), ext="GAIA", columns=["SOURCE_ID"])
    if len(mws) != len(gaia):
        raise ValueError("MWS HDUs are not row aligned")
    pos = np.searchsorted(ids, gaia["SOURCE_ID"])
    safe = np.minimum(pos, len(ids) - 1)
    member = ((pos < len(ids)) & (ids[safe] == gaia["SOURCE_ID"]))
    group = np.zeros(len(mws), np.int64)
    group[member] = member_groups[safe[member]]
    good = (member & mws["PRIMARY"]
            & (np.char.strip(mws["RR_SPECTYPE"].astype("U")) == "STAR")
            & (mws["RVS_WARN"] == 0) & mws["SUCCESS"]
            & np.isfinite(mws["SN_R"]) & (mws["SN_R"] >= 10)
            & np.isfinite(mws["VSINI"]) & (mws["VSINI"] >= 0)
            & (mws["VSINI"] < 30) & np.isfinite(mws["TEFF"])
            & (mws["TEFF"] > 0) & (mws["TEFF"] < 7000)
            & np.isfinite(mws["TEFF_ERR"]) & (mws["TEFF_ERR"] > 0)
            & np.isfinite(mws["LOGG"]) & np.isfinite(mws["LOGG_ERR"])
            & (mws["LOGG_ERR"] > 0)
            & (mws["LOGG"] + mws["LOGG_ERR"] < 3.2))
    result = []
    for group_id in np.unique(group[good]):
        if group_id < 100000:
            continue
        chosen = np.flatnonzero(good & (group == group_id))
        gc = catalog[int(group_id) - 100000]
        source_ids = np.unique(gaia["SOURCE_ID"][chosen])
        result.append({"group": int(group_id), "name": str(gc["Name"]),
                       "qualified_giant_spectra": len(chosen),
                       "unique_gaia_members": len(source_ids),
                       "cool_lt5200": int(np.sum(mws["TEFF"][chosen] < 5200)),
                       "ra": float(gc["ra"]), "dec": float(gc["dec"]),
                       "distance_kpc": float(gc["dist"]),
                       "distance_error_kpc": float(gc["disterr"]),
                       "member_source_id_examples": source_ids[:10].astype(int).tolist()})
    result.sort(key=lambda item: (-item["cool_lt5200"], -item["unique_gaia_members"]))
    write_json(OUT, {"source": str(MWS), "reference": str(REF),
                     "membership_definition": "VasilievBaumgardt2021 memberprob>=0.5, direct archive, ambiguous cross-cluster Gaia IDs dropped",
                     "ambiguous_membership_rows_dropped": ambiguous,
                     "qualified_giant_members": int(good.sum()),
                     "systems": result, "compute_host": socket.gethostname()})
    print(json.dumps({"systems": len(result), "top": result[:5]}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose sparse exact-target overlap with DESI distance VACs."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.stage1 import require_compute, write_json
from giant_vac_inventory import exact_pair_join

GIANTS = Path("data/giant_vac_v1/same_star.fits")
SPEC = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-specdis/v2.1/iron-yr1-v2.1.fits")
SP = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-spdist/v1.0/mws-iron-SPDist-240613.fits")
OUT = Path("manifests/giant_vac_identity_audit_v2.json")


def match_one(query, catalog):
    order = np.argsort(catalog, kind="stable")
    value = catalog[order]
    location = np.searchsorted(value, query)
    safe = np.minimum(location, len(value) - 1)
    found = (location < len(value)) & (value[safe] == query)
    before = np.maximum(safe - 1, 0)
    after = np.minimum(safe + 1, len(value) - 1)
    unique = ((safe == 0) | (value[before] != query)) & (
        (safe == len(value)-1) | (value[after] != query))
    return np.where(found, order[safe], -1), found & unique


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    giant = fitsio.read(str(GIANTS), columns=["source_id", "desi_object_id",
                        "clean_low_dust", "desi_teff", "desi_snr"])
    target = np.char.strip(giant["desi_object_id"].astype("U")).astype(np.int64)
    result = {"giants": len(giant), "clean": int(giant["clean_low_dust"].sum()),
              "catalogs": {}, "compute_host": socket.gethostname()}
    for name, path, ext in (("specdis", SPEC, "SPECDIS"), ("spdist", SP, "SPDIST")):
        catalog = fitsio.read(str(path), ext=ext, columns=["TARGETID", "SOURCE_ID"])
        by_target, target_unique = match_one(target, catalog["TARGETID"])
        by_source, source_unique = match_one(giant["source_id"], catalog["SOURCE_ID"])
        target_found, source_found = by_target >= 0, by_source >= 0
        both = target_found & (catalog["SOURCE_ID"][np.maximum(by_target, 0)] == giant["source_id"])
        joined = exact_pair_join(target, giant["source_id"], catalog)
        pair_joined = joined >= 0
        disagreement = np.flatnonzero(pair_joined != both)
        sid_only = source_found & source_unique & ~both & (giant["source_id"] > 0)
        examples = []
        for i in np.flatnonzero(sid_only)[:8]:
            j = by_source[i]
            examples.append({"source_id": int(giant["source_id"][i]),
                             "joint_targetid": int(target[i]),
                             "vac_targetid": int(catalog["TARGETID"][j]),
                             "teff": float(giant["desi_teff"][i]),
                             "snr": float(giant["desi_snr"][i])})
        result["catalogs"][name] = {
            "catalog_rows": len(catalog), "target_found": int(target_found.sum()),
            "target_unique": int((target_found & target_unique).sum()),
            "source_found": int((source_found & (giant["source_id"] > 0)).sum()),
            "source_unique": int((source_found & source_unique & (giant["source_id"] > 0)).sum()),
            "exact_pair": int(both.sum()),
            "original_exact_pair_join": int(pair_joined.sum()),
            "join_disagreement_examples": [{"targetid": int(target[i]),
                 "source_id": int(giant["source_id"][i]),
                 "target_match_index": int(by_target[i]),
                 "original_join_index": int(joined[i])}
                for i in disagreement[:8]],
            "unique_source_different_target": int(sid_only.sum()),
            "unique_source_different_target_clean": int((sid_only & giant["clean_low_dust"]).sum()),
            "source_only_examples": examples}
    write_json(OUT, result)
    print(json.dumps(result["catalogs"]), flush=True)


if __name__ == "__main__":
    main()

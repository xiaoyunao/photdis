#!/usr/bin/env python3
"""Extract the preregistered NGC 5466 blind cluster after the fit is frozen."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps"), str(ROOT / "scripts")]
import fitsio
import numpy as np

from cluster_rgb_inventory import MEMBERS, matched_sweeps, read_members, separation_arcmin
from cluster_rgb_phase import classify
from photdis.stage1 import legacy_ids, require_compute, structured, write_json


def main():
    require_compute()
    frozen = Path("manifests/cluster_rgb_conditional_model_v1.json")
    if not frozen.exists():
        raise RuntimeError("Fit must be frozen before NGC 5466 extraction")
    registered = json.loads(Path("configs/cluster_rgb_conditional_v1.json").read_text())
    cfg = registered["whole_cluster_test"]["ngc5466"]
    phase_cfg = json.loads(Path("configs/cluster_rgb_phase_regions_v1.json").read_text())
    screen = phase_cfg["systems"]["m53"]
    out = Path("data/cluster_rgb_gr_v2/ngc5466.fits")
    manifest = Path("manifests/cluster_rgb_ngc5466_blind_coverage_v1.json")
    if out.exists() or manifest.exists():
        raise FileExistsError(out if out.exists() else manifest)
    members = read_members(MEMBERS, cfg["member_file"])
    eligible = members[members["pmember"] >= .5]
    rows, photo, raw_count = matched_sweeps(cfg, eligible["source_id"], prefer="gr")
    source = rows["REF_ID"].astype(np.int64)
    order = np.argsort(eligible["source_id"])
    eligible = eligible[order]
    index = np.searchsorted(eligible["source_id"], source)
    if np.any(index >= len(eligible)) or np.any(eligible["source_id"][index] != source):
        raise ValueError("NGC 5466 Gaia identity mismatch")
    member = eligible[index]
    phase = classify(member["gaia_g"]-cfg["external_mu"], member["bp_rp"], screen)
    table = structured({
        "source_id": source, "ls_id": legacy_ids(rows), "release": rows["RELEASE"],
        "phase": phase, "pmember": member["pmember"],
        "gaia_g": member["gaia_g"], "bp_rp": member["bp_rp"],
        "radius_arcmin": separation_arcmin(rows["RA"],rows["DEC"],cfg["center"]),
        "ebv": rows["EBV"], "g0": photo["g0"], "r0": photo["r0"],
        "z0": photo["z0"], "gr0": photo["gr0"], "rz0": photo["rz0"],
        "sigma_g": photo["sigma_g"], "sigma_r": photo["sigma_r"],
        "sigma_z": photo["sigma_z"],
        "gr_base": photo["gr_base"], "gr_clean": photo["gr_clean"],
        "grz_clean_same_row": photo["grz_clean"], "maskbits": rows["MASKBITS"],
        **{key.lower(): rows[key] for key in rows.dtype.names
           if key.startswith(("FRACMASKED_", "FRACIN_", "FRACFLUX_", "RCHISQ_"))
           and key.endswith(("_G", "_R", "_Z"))},
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(out), table)
    high = table["pmember"] > .9
    phase = table["phase"].astype("U")
    counts = {}
    for stage in ("upper_rgb", "lower_rgb", "hb", "possible_agb", "ambiguous"):
        selected = high & (phase == stage)
        counts[stage] = {"members": int(selected.sum()),
                         "gr_clean": int((selected & table["gr_clean"]).sum()),
                         "grz_clean": int((selected & table["grz_clean_same_row"]).sum())}
    result = {"role": "registered_whole_cluster_test",
              "registration": "configs/cluster_rgb_conditional_v1.json",
              "frozen_model": str(frozen), "member_source": str(MEMBERS),
              "raw_match_count": raw_count, "unique_members": len(table),
              "release_counts": {str(v): int((table["release"] == v).sum())
                                 for v in np.unique(table["release"])},
              "phase_counts": counts, "data_file": str(out),
              "compute_host": socket.gethostname()}
    write_json(manifest, result)
    print(json.dumps({"unique_members":len(table),"counts":counts}), flush=True)


if __name__ == "__main__":
    main()

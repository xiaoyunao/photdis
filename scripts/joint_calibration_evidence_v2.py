#!/usr/bin/env python3
"""Reconcile RVS, Redrock and Gaia CMD evidence in an existing joint sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.class_evidence import desi_class_evidence_v3
from photdis.stage1 import require_compute, write_json
from cold_v4 import sha

EXTRA = [("class_evidence_available", "?"), ("rvs_parameter_quality", "?"),
         ("blue_training_eligible", "?"),
         ("abundance_calibration_valid", "?"), ("redrock_type", "S8"),
         ("redrock_quality_state", "S28"),
         ("redrock_nonstellar_evidence", "?"),
         ("cmd_spectral_conflict", "?"),
         ("trusted_blue_core", "?")]
UPDATED = ("class_evidence", "class_evidence_source",
           "class_evidence_quality", "class_conflicts_cold_dwarf")


def summarize(rows):
    labels = np.char.strip(rows["class_evidence"].astype("U"))
    layers = np.char.strip(rows["parallax_layer"].astype("U"))
    core = rows["trusted_blue_core"]
    poor = core & np.isfinite(rows["feh_cal"]) & (rows["feh_cal"] < -1.5)
    giant = labels == "conservative_giant"
    teff = rows["desi_teff"]
    result = {"rows": len(rows),
              "class_counts": {label: int(np.sum(labels == label))
                               for label in np.unique(labels)},
              "redrock_nonstellar_raw": int(rows["redrock_nonstellar_evidence"].sum()),
              "cmd_wd_candidate": int(rows["wd_cmd_candidate"].sum()),
              "cmd_spectral_conflict": int(rows["cmd_spectral_conflict"].sum()),
              "trusted_blue_core_clean_low_dust": int(core.sum()),
              "trusted_blue_core_with_parallax": int(np.sum(core & (layers != "unusable"))),
              "trusted_poor_core_parallax_layers": {
                  layer: int(np.sum(poor & (layers == layer))) for layer in
                  ("high_ge20", "mid_10_20", "weak_0_10", "negative", "unusable")},
              "giants": {"all": int(giant.sum()),
                         "clean_low_dust": int(np.sum(giant & rows["clean_low_dust"])),
                         "high_snr": int(np.sum(giant & (layers == "high_ge20"))),
                         "feh_cal_valid": int(np.sum(giant & rows["abundance_calibration_valid"])),
                         "teff_lt_4200": int(np.sum(giant & (teff < 4200))),
                         "teff_4200_5000": int(np.sum(giant & (teff >= 4200) & (teff < 5000))),
                         "teff_5000_7000": int(np.sum(giant & (teff >= 5000) & (teff < 7000))),
                         "logg_lt_1p8": int(np.sum(giant & (rows["desi_logg"] < 1.8)))}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Joint evidence v2 output already exists")
    prior = fitsio.read(str(args.source))
    evidence = desi_class_evidence_v3({name: prior[name] for name in prior.dtype.names})
    result = np.empty(len(prior), dtype=prior.dtype.descr + EXTRA)
    for name in prior.dtype.names:
        result[name] = prior[name]
    for name in UPDATED + tuple(name for name, _ in EXTRA if name in evidence):
        result[name] = evidence[name]
    labels = np.char.strip(result["class_evidence"].astype("U"))
    conflict = result["wd_cmd_candidate"] & np.isin(
        labels, ["conservative_dwarf", "conservative_giant"])
    result["cmd_spectral_conflict"] = conflict
    result["trusted_blue_core"] = (result["blue_training_eligible"]
                                   & result["clean_low_dust"] & ~conflict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(args.output), result, extname="JOINT", clobber=False)
    roles = np.char.strip(result["role"].astype("U"))
    fields = np.char.strip(result["field"].astype("U"))
    manifest = {"source": str(args.source), "source_sha256": sha(args.source),
                "output": str(args.output), "output_sha256": sha(args.output),
                "all": summarize(result),
                "by_role": {role: summarize(result[roles == role])
                            for role in ("development", "reserved_validation")},
                "by_field": {field: summarize(result[fields == field])
                             for field in np.unique(fields)},
                "compute_host": socket.gethostname()}
    write_json(args.manifest, manifest)
    print(json.dumps({"all": manifest["all"], "by_role": manifest["by_role"]}), flush=True)


if __name__ == "__main__":
    main()

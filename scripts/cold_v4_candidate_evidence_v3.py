#!/usr/bin/env python3
"""Version frozen-v4 candidates with general RVS and raw Redrock evidence."""
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
from cold_v4_candidate_export import FIELDS

COLUMNS = ["ls_id", "desi_match", "desi_match_ambiguous", "desi_spectype",
           "desi_quality_reason", "desi_teff", "desi_teff_err", "desi_logg",
           "desi_logg_err", "desi_feh", "desi_feh_err"]
EXTRA = [("rvs_parameter_quality", "?"), ("blue_training_eligible", "?"),
         ("abundance_calibration_valid", "?"), ("redrock_type", "S8"),
         ("redrock_quality_state", "S28"),
         ("redrock_nonstellar_evidence", "?")]
UPDATED = ("class_evidence", "class_evidence_source",
           "class_evidence_quality", "class_evidence_available",
           "class_conflicts_cold_dwarf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True, choices=[
        json.loads(Path(path).read_text())["name"] for path in FIELDS])
    parser.add_argument("--chunk", type=int, default=100000)
    args = parser.parse_args()
    require_compute()
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    field_config = json.loads(Path(next(path for path in FIELDS if
        json.loads(Path(path).read_text())["name"] == args.field)).read_text())
    source = Path(field_config["output_dir"]) / "field_diagnostics.fits"
    prior = Path("data/cold_v4_candidates_v2") / f"{args.field}.fits"
    target = Path("data/cold_v4_candidates_v3") / f"{args.field}.fits"
    manifest = Path("manifests") / f"cold_v4_candidates_v3_{args.field}.json"
    if target.exists() or manifest.exists():
        raise FileExistsError("Candidate v3 output already exists")
    old = fitsio.read(str(prior))
    if len(np.unique(old["ls_id"])) != len(old):
        raise ValueError("Candidate Legacy identities are not unique")
    result = np.empty(len(old), dtype=old.dtype.descr + EXTRA)
    for name in old.dtype.names:
        result[name] = old[name]
    order = np.argsort(old["ls_id"])
    sorted_ids = old["ls_id"][order]
    seen = np.zeros(len(old), bool)
    with fitsio.FITS(str(source)) as file:
        nrows = file[1].get_nrows()
        for start in range(0, nrows, args.chunk):
            raw = file[1].read(columns=COLUMNS,
                               rows=np.arange(start, min(start + args.chunk, nrows)))
            pos = np.searchsorted(sorted_ids, raw["ls_id"])
            hit = (pos < len(sorted_ids)) & (sorted_ids[np.minimum(
                pos, len(sorted_ids) - 1)] == raw["ls_id"])
            if not hit.any():
                continue
            destination = order[pos[hit]]
            if seen[destination].any():
                raise ValueError("Repeated source Legacy identity")
            evidence = desi_class_evidence_v3({name: raw[name][hit]
                                              for name in raw.dtype.names})
            for name in UPDATED + tuple(name for name, _ in EXTRA):
                result[name][destination] = evidence[name]
            seen[destination] = True
    if not seen.all():
        raise ValueError(f"Missing diagnostic rows: {int((~seen).sum())}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(target), result, extname="CANDIDATES", clobber=False)
    labels = np.char.strip(result["class_evidence"].astype("U"))
    counts = {label: int(np.sum(labels == label)) for label in np.unique(labels)}
    summary = {"field": args.field, "input": str(prior), "input_sha256": sha(prior),
               "source": str(source), "output": str(target),
               "output_sha256": sha(target), "rows": len(result),
               "physical_class_counts": counts,
               "cold_dwarf_conflicts": int(result["class_conflicts_cold_dwarf"].sum()),
               "raw_redrock_nonstellar": int(result["redrock_nonstellar_evidence"].sum()),
               "blue_training_eligible": int(result["blue_training_eligible"].sum()),
               "abundance_calibration_valid": int(result["abundance_calibration_valid"].sum()),
               "compute_host": socket.gethostname()}
    write_json(manifest, summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

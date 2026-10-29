#!/usr/bin/env python3
"""Check every v2 candidate against the frozen v1 catalog on allocated CPU."""
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
from cold_v4_candidate_export import FIELDS

OUTPUT = Path("manifests/cold_v4_candidate_v2_validation.json")
CHECK = ("ls_id", "source_id", "tier", "mg_median", "mu_median",
         "distance_pc_median", "reason_b", "reason_c",
         "support_ratio_a", "support_ratio_b", "support_ratio_c",
         "infrared_fallback_conflict", "used_tier_high_quality", "depth_state")


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = {"fields": {}, "all_rows": 0, "all_evidence_available": 0,
              "all_conflicts": 0, "compute_host": socket.gethostname()}
    for config in FIELDS:
        field = json.loads(Path(config).read_text())["name"]
        old_path = Path("data/cold_v4_candidates") / f"{field}.fits"
        new_path = Path("data/cold_v4_candidates_v2") / f"{field}.fits"
        with fitsio.FITS(str(old_path)) as old_file, fitsio.FITS(str(new_path)) as new_file:
            assert len(old_file) == len(new_file) == 2
            assert old_file[1].get_nrows() == new_file[1].get_nrows()
            old = old_file[1].read(columns=list(CHECK))
            new = new_file[1].read(columns=list(CHECK) + ["class_evidence",
                "class_evidence_available", "class_conflicts_cold_dwarf"])
        for name in CHECK:
            same = np.array_equal(old[name], new[name],
                                  equal_nan=old.dtype[name].kind == "f")
            if not same:
                raise ValueError(f"Frozen v1/v2 mismatch in {field}: {name}")
        if not np.array_equal(new["class_evidence_available"],
                              new["class_evidence"] != b"unknown"):
            raise ValueError(f"Evidence availability mismatch in {field}")
        conflicts = np.isin(new["class_evidence"], [b"cool_giant", b"nonstellar"])
        if not np.array_equal(new["class_conflicts_cold_dwarf"], conflicts):
            raise ValueError(f"Conflict flag mismatch in {field}")
        counts = {label.decode(): int(np.sum(new["class_evidence"] == label))
                  for label in np.unique(new["class_evidence"])}
        entry = {"rows": len(new), "frozen_columns_equal": len(CHECK),
                 "class_counts": counts,
                 "evidence_available": int(new["class_evidence_available"].sum()),
                 "class_conflicts_cold_dwarf": int(conflicts.sum())}
        report["fields"][field] = entry
        report["all_rows"] += len(new)
        report["all_evidence_available"] += entry["evidence_available"]
        report["all_conflicts"] += entry["class_conflicts_cold_dwarf"]
        print(json.dumps({"field": field, **entry}), flush=True)
    write_json(OUTPUT, report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify every frozen v2 value survives the v3 evidence-only update."""
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

OUTPUT = Path("manifests/cold_v4_candidate_v3_validation.json")


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    summary = {"fields": {}, "all_rows": 0, "all_old_columns_equal": True,
               "compute_host": socket.gethostname()}
    for field_file in FIELDS:
        field = json.loads(Path(field_file).read_text())["name"]
        old = fitsio.read(str(Path("data/cold_v4_candidates_v2") / f"{field}.fits"))
        new = fitsio.read(str(Path("data/cold_v4_candidates_v3") / f"{field}.fits"))
        if len(old) != len(new):
            raise ValueError(f"Row count differs: {field}")
        changed = ("class_evidence", "class_evidence_source",
                   "class_evidence_quality", "class_evidence_available",
                   "class_conflicts_cold_dwarf")
        for name in old.dtype.names:
            if name in changed:
                continue
            if not np.array_equal(old[name], new[name],
                                  equal_nan=old.dtype[name].kind == "f"):
                raise ValueError(f"Frozen v2 value differs: {field}, {name}")
        labels = np.char.strip(new["class_evidence"].astype("U"))
        conflict = labels == "conservative_giant"
        if not np.array_equal(conflict, new["class_conflicts_cold_dwarf"]):
            raise ValueError(f"Giant conflict flag differs: {field}")
        rr = new["redrock_nonstellar_evidence"]
        if np.any(rr & (np.char.strip(new["redrock_quality_state"].astype("U"))
                         != "unassessed_no_rr_quality")):
            raise ValueError(f"Redrock quality claim differs: {field}")
        entry = {"rows": len(new), "preserved_v2_columns": len(old.dtype.names) - len(changed),
                 "class_counts": {label: int(np.sum(labels == label))
                                  for label in np.unique(labels)},
                 "giant_conflicts": int(conflict.sum()),
                 "raw_redrock_nonstellar": int(rr.sum()),
                 "blue_training_eligible": int(new["blue_training_eligible"].sum())}
        summary["fields"][field] = entry
        summary["all_rows"] += len(new)
        print(json.dumps({"field": field, **entry}), flush=True)
    write_json(OUTPUT, summary)


if __name__ == "__main__":
    main()

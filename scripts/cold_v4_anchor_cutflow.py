#!/usr/bin/env python3
"""Diagnose which frozen cold-anchor condition removes a registered field."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import numpy as np

from photdis.followup import common_nonoptical
from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.stage1 import require_compute, write_json
from cold_v4 import configured_field

OUTPUT = Path("data/cold_v4_native_transfer/anchor_cutflow_development.json")
FIELDS = ("configs/field_native_ra060_070.json",
          "configs/field_native_ra235_245.json",
          "configs/field_native_ra180_190.json")


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    references = load_reference_catalogs("data/reference")
    result = {}
    for path in FIELDS:
        table, cold, mg, meta = configured_field(path, references)
        external = annotate_holdouts(references, table["source_id"],
                                     table["ra"], table["dec"])["external_any"]
        _, flags, _ = common_nonoptical(table, external)
        cumulative = np.ones(len(mg), bool)
        cutflow = []
        for name, passed in flags.items():
            cumulative &= passed
            cutflow.append({"name": name, "individual_pass": int(passed.sum()),
                            "cumulative_pass": int(cumulative.sum())})
        color = np.asarray(table["bp_rp_checked"], float)
        valid = np.isfinite(color)
        result[meta["name"]] = {"strict_anchors": len(mg),
                                "cold_anchors": int(cold.sum()),
                                "gaia_color_finite": int(valid.sum()),
                                "gaia_color_1p3_to_3p5": int(np.sum(
                                    valid & (color >= 1.3) & (color < 3.5))),
                                "mg_gt_5": int(np.sum(np.isfinite(mg) & (mg > 5))),
                                "external_any": int(external.sum()),
                                "cutflow": cutflow}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, result)


if __name__ == "__main__":
    main()

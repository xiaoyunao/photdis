#!/usr/bin/env python3
"""Audit a fixed photometric A/B/C routing rule on untouched final fields."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import joblib
import numpy as np

from photdis.candidate import flags
from photdis.holdouts import load_reference_catalogs
from photdis.quantile import metrics
from photdis.stage1 import require_compute, write_json
from cold_candidate_demo import sha, TIERS
from conditional_calibration import load_field, sky_block_ci
from target_coverage import fixed_rule

OUTPUT = Path("manifests/selected_tier_validation_v1.json")
FIELDS = ("final_north", "final_south", "supp_north", "supp_south")
RED_BINS = ((1.3, 1.8), (1.8, 2.3), (2.3, 2.8), (2.8, 3.5))


def score(truth, prediction, g0, parallax, ra, dec):
    return {**metrics(truth, prediction, g0, parallax),
            "sky_block_ci": sky_block_ci(truth, prediction, ra, dec)}


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = json.loads(Path("configs/followup_fields.json").read_text())
    if sha("manifests/cold_baseline.json") != frozen["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline manifest changed")
    rule, tree = fixed_rule()
    published_rule = json.loads(Path("manifests/target_coverage_v2.json").read_text())[
        "fixed_from_original_non_test_cold_anchors"]
    if rule != published_rule:
        raise ValueError("Published color/support rule changed")
    forests = {}
    for level, (name, _) in TIERS.items():
        path = Path("data/cold_baseline_v3") / f"{name}.joblib"
        if sha(path) != frozen["baseline_model_sha256"][name]:
            raise ValueError(f"Frozen model changed: {name}")
        forests[level] = joblib.load(path)

    references = load_reference_catalogs("data/reference")
    summary = {"rule": "grz color box plus tenth-neighbor support; C if available, else B, else A",
               "role": "direct frozen-model final-field evaluation; no refitting",
               "fields": {}, "compute_host": socket.gethostname()}
    for field_name in FIELDS:
        table, cold, mg, field = load_field(field_name, references)
        visible = {key: table[key] for key in
                   ("gr0", "rz0", "grz_base", "grz_w1", "grz_w1_w2", "r")}
        visible["grz_high_quality"] = table["anchor"]
        selection = flags(visible, rule, tree)
        take = cold & selection["color_box"] & selection["support_ok"]
        prediction = np.full((len(table["source_id"]), 5), np.nan)
        by_tier = {}
        for level, (name, features) in TIERS.items():
            subset = take & (selection["tier"] == level)
            count = int(subset.sum())
            by_tier[name] = count
            if count:
                matrix = np.column_stack([table[key][subset] for key in features])
                if not np.isfinite(matrix).all():
                    raise ValueError(f"Nonfinite {name} features in {field_name}")
                prediction[subset] = forests[level].predict(matrix)
        if not int(take.sum()):
            raise ValueError(f"No selected cold anchors in {field_name}")
        result = {"field": field, "strict_cold_rows": int(cold.sum()),
                  "color_box_rows": int((cold & selection["color_box"]).sum()),
                  "box_and_support_rows": int(take.sum()),
                  "chosen_tier_counts": by_tier,
                  "direct_final_distance": score(mg[take], prediction[take],
                                                 table["g0"][take], table["p_corr"][take],
                                                 table["ra"][take], table["dec"][take]),
                  "Gaia_color_diagnostics_only": []}
        for lo, hi in RED_BINS:
            subset = take & (table["bp_rp_checked"] >= lo) & (table["bp_rp_checked"] < hi)
            if subset.sum():
                result["Gaia_color_diagnostics_only"].append({
                    "BP_RP": [lo, hi],
                    "metrics": score(mg[subset], prediction[subset], table["g0"][subset],
                                     table["p_corr"][subset], table["ra"][subset],
                                     table["dec"][subset])})
        summary["fields"][field_name] = result
        print(json.dumps({"field": field_name,
                          "selected": result["box_and_support_rows"],
                          "tier_counts": by_tier,
                          "metrics": result["direct_final_distance"]}), flush=True)
    write_json(OUTPUT, summary)


if __name__ == "__main__":
    main()

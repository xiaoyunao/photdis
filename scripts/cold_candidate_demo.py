#!/usr/bin/env python3
"""Export a bounded photometry-only cold-dwarf-hypothesis distance sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import joblib
import numpy as np

from photdis.candidate import flags
from photdis.stage1 import require_compute, write_json
from target_coverage import fixed_rule

SOURCE = Path("data/field_final_ra050_060_v1/field_diagnostics.fits")
OUTPUT = Path("data/cold_candidate_demo_v1/cold_dwarf_hypothesis.fits")
MANIFEST = Path("manifests/cold_candidate_demo_v1.json")
TIERS = {1: ("A_grz", ("gr0", "rz0")),
         2: ("B_grzW1", ("gr0", "rz0", "zw10")),
         3: ("C_grzW1W2", ("gr0", "rz0", "zw10", "w1w20"))}
COLUMNS = sorted(set(("source_id", "ra", "dec", "g0", "r", "grz_base",
                      "grz_w1", "grz_w1_w2", "grz_high_quality",
                      *{feature for _, features in TIERS.values() for feature in features})))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_sample(table, selection, max_rows, seed):
    eligible = np.flatnonzero(selection["color_box"])
    if not len(eligible):
        raise ValueError("No color-box candidates")
    if len(eligible) > max_rows:
        eligible = np.random.default_rng(seed).choice(eligible, max_rows, replace=False)
    return np.sort(eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    if args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    require_compute()
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("Demo products already exist")

    frozen = json.loads(Path("configs/followup_fields.json").read_text())
    if sha("manifests/cold_baseline.json") != frozen["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline manifest changed")
    rule, tree = fixed_rule()
    published_rule = json.loads(Path("manifests/target_coverage_v2.json").read_text())[
        "fixed_from_original_non_test_cold_anchors"]
    if rule != published_rule:
        raise ValueError("Published color/support rule changed")

    table = fitsio.read(SOURCE, columns=COLUMNS)
    selection = flags(table, rule, tree)
    rows = select_sample(table, selection, args.max_rows, args.seed)
    dtype = [("source_id", table["source_id"].dtype), ("ra", "f8"),
             ("dec", "f8"), ("g0", "f4"), ("r", "f4"),
             ("tier", "u1"), ("support_ok", "?"),
             ("strict_optical", "?"), ("r_ge_20", "?"),
             ("type_unverified", "?"), ("mg_median", "f4"),
             ("distance_pc", "f4")]
    result = np.empty(len(rows), dtype=dtype)
    for name in ("source_id", "ra", "dec", "g0", "r"):
        result[name] = table[name][rows]
    for name in ("tier", "support_ok", "strict_optical", "r_ge_20", "type_unverified"):
        result[name] = selection[name][rows]
    result["mg_median"] = np.nan
    for level, (name, features) in TIERS.items():
        local = np.flatnonzero(result["tier"] == level)
        if not len(local):
            continue
        model_path = Path("data/cold_baseline_v3") / f"{name}.joblib"
        if sha(model_path) != frozen["baseline_model_sha256"][name]:
            raise ValueError(f"Frozen model changed: {name}")
        model = joblib.load(model_path)
        matrix = np.column_stack([table[feature][rows[local]] for feature in features])
        if not np.isfinite(matrix).all():
            raise ValueError(f"Nonfinite {name} features in selected sample")
        result["mg_median"][local] = model.predict(matrix)[:, 2]
    result["distance_pc"] = 10 ** ((result["g0"] - result["mg_median"] + 5) / 5)
    if not np.isfinite(result["distance_pc"]).all():
        raise ValueError("Nonfinite predicted distances")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(OUTPUT, result)
    counts = {name: int(np.sum(result["tier"] == level)) for level, (name, _) in TIERS.items()}
    manifest = {"role": "photometry_only_cold_dwarf_hypothesis_demo",
                "field": "final_ra050_060_dec_m030_m020",
                "sample_rule": "uniform without replacement among color-box candidates",
                "seed": args.seed, "max_rows": args.max_rows,
                "mother_color_box_rows": int(selection["color_box"].sum()),
                "output_rows": len(result), "tier_counts": counts,
                "support_ok_rows": int(result["support_ok"].sum()),
                "strict_optical_rows": int(result["strict_optical"].sum()),
                "r_ge_20_rows": int(result["r_ge_20"].sum()),
                "type_risk": "All rows are unverified cold-dwarf hypotheses; no class probability is assigned",
                "distance_scope": "Frozen baseline median only; no published 68/95% interval",
                "frozen_baseline_commit": frozen["baseline_commit"],
                "frozen_model_sha256": frozen["baseline_model_sha256"],
                "output_sha256": sha(OUTPUT), "compute_host": socket.gethostname()}
    write_json(MANIFEST, manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

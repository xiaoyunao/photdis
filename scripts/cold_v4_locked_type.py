#!/usr/bin/env python3
"""DESI class-conditional selection after the frozen clean WISE20 route."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import joblib
import numpy as np

from photdis.audit import rvs_metallicity
from photdis.route import LEVEL, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import CAL_CONFIG, FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED, sha
from cold_v4_type_audit import COLS, class_masks, unique_spectra, wilson

LOCK = Path("configs/cold_v4_deep_policy.json")
RESULT = Path("data/cold_v4_deep/locked_type.json")


def class_result(mask, baseline, chosen):
    n = int(mask.sum())
    old = baseline["tier"]
    selected = mask & (chosen > 0)
    count = int(selected.sum())
    return {"n_labeled_in_grz_base": n,
            "baseline_route_selected": int(np.sum(mask & baseline["selected"])),
            "locked_clean_route_selected": count,
            "locked_rate_given_label": count / n if n else None,
            "locked_rate_wilson95": wilson(count, n),
            "by_chosen_tier": {tier: int(np.sum(mask & (chosen == level)))
                               for tier, level in LEVEL.items()},
            "higher_tier_replaced_or_removed": int(np.sum(mask & (old > chosen) & (old > 0))),
            "baseline_selected_lost": int(np.sum(mask & baseline["selected"] & (chosen == 0)))}


def field_result(config, route, policy):
    task = json.loads(Path(config).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    columns = sorted(set(COLS + ["snr_w1", "snr_w2"]))
    raw = fitsio.read(path, columns=columns)
    table = {name: raw[name] for name in raw.dtype.names}
    unique, dedup = unique_spectra(table)
    labels = class_masks(table, unique & table["grz_base"])
    baseline = route_rows(table, route)
    chosen = choose_policy(table, baseline, policy)
    feh, valid = rvs_metallicity(table["desi_feh"], table["desi_teff"],
                                 table["desi_logg"])
    poor = (labels["cool_dwarf"] & valid &
            np.isfinite(table["desi_feh_err"]) & (table["desi_feh_err"] > 0) &
            (feh < -1.5))
    return {"field": task["name"], "deduplication": dedup,
            "label_scope": "unambiguous DESI object, quality-qualified where required, grz_base; class-conditional rates not candidate purity",
            "labels": {name: class_result(mask, baseline, chosen)
                       for name, mask in labels.items()},
            "calibrated_domain_feh_below_minus_1p5_cool_dwarf": class_result(
                poor, baseline, chosen)}


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen preparation changed")
    lock = json.loads(LOCK.read_text())
    if lock["prepared_sha256"] != sha(PREPARED):
        raise ValueError("Locked policy differs from prepared v4")
    prepared = json.loads(PREPARED.read_text())
    route_path = OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen route changed")
    route = joblib.load(route_path)
    result = {"prepared_sha256": sha(PREPARED),
              "policy_lock_sha256": sha(LOCK),
              "compute_host": socket.gethostname(),
              "class_conditional_not_purity": True,
              "fields": {}}
    for config in (CAL_CONFIG, *FINAL_CONFIGS):
        row = field_result(config, route, lock["chosen_policy"])
        result["fields"][row["field"]] = row
        print(json.dumps({"field": row["field"], "locked_cool_dwarf":
            row["labels"]["cool_dwarf"]["locked_clean_route_selected"]}), flush=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, result)


if __name__ == "__main__":
    main()

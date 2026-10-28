#!/usr/bin/env python3
"""Compare wide binaries and cluster scatter under old and locked WISE20 routing."""
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

from photdis.holdouts import load_reference_catalogs
from photdis.route import FEATURES, LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED, sha
from cold_v4_system_validation import (READ, component_median_ci,
                                       evaluate_clusters, evaluate_pairs,
                                       robust_summary)

LOCK = Path("configs/cold_v4_deep_policy.json")
RESULT = Path("data/cold_v4_systems/relative_validation_v3.json")
PAIR_CATALOG = "data/reference/wide_binary_pairs_v1.npz"


def member_rows(config_path, references):
    task = json.loads(Path(config_path).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    with fitsio.FITS(str(path)) as file:
        ids = file[1].read(columns=["source_id"])["source_id"]
        external = np.isin(ids, references.binary_source_id) | np.isin(
            ids, references.member_source_id)
        index = np.flatnonzero(external)
        if not len(index):
            return None, {"field": task["name"], "reference_rows": 0}
        columns = sorted(set(READ + ["snr_w1", "snr_w2", "sigma_g", "sigma_r",
                                     "sigma_z", "sigma_w1", "sigma_w2"]))
        raw = file[1].read(columns=columns, rows=index)
    _, inverse, count = np.unique(raw["source_id"], return_inverse=True,
                                  return_counts=True)
    unique = count[inverse] == 1
    raw = raw[unique]
    return {name: raw[name] for name in raw.dtype.names}, {
        "field": task["name"], "reference_rows": len(index),
        "unique_Gaia_rows": len(raw),
        "duplicate_identity_rows_excluded": int(np.sum(~unique))}


def prediction_records(table, levels, forests):
    selected = levels > 0
    if not selected.any():
        return np.empty(0, dtype=[("source_id", "i8"), ("r0", "f8"),
                                  ("rz0", "f8"), ("mu_median", "f8")])
    output = np.empty(int(selected.sum()), dtype=[("source_id", "i8"),
                                                  ("r0", "f8"), ("rz0", "f8"),
                                                  ("mu_median", "f8")])
    for name in ("source_id", "r0", "rz0"):
        output[name] = table[name][selected]
    mu = np.full(len(selected), np.nan)
    for tier, number in LEVEL.items():
        use = levels == number
        if use.any():
            predicted = forests[tier].predict(feature_matrix(table, tier)[use])
            mu[use] = np.asarray(table["g0"][use], float) - predicted[:, 2]
    output["mu_median"] = mu[selected]
    if not np.isfinite(output["mu_median"]).all():
        raise ValueError("Nonfinite system distance modulus")
    return output


def paired_binary_change(old, new, pair_catalog):
    old_order = np.argsort(old["source_id"])
    new_order = np.argsort(new["source_id"])
    old_ids, new_ids = old["source_id"][old_order], new["source_id"][new_order]
    common = np.intersect1d(old_ids, new_ids)
    if not len(common):
        return {"n_pairs": 0}
    old_common = old[old_order[np.searchsorted(old_ids, common)]]
    new_common = new[new_order[np.searchsorted(new_ids, common)]]
    with np.load(pair_catalog, allow_pickle=False) as catalog:
        left = np.asarray(catalog["source_id1"], np.int64)
        right = np.asarray(catalog["source_id2"], np.int64)
        group = np.asarray(catalog["group"], np.int64)
    a, b = np.searchsorted(common, left), np.searchsorted(common, right)
    found = (a < len(common)) & (b < len(common))
    found[found] &= (common[a[found]] == left[found]) & (
        common[b[found]] == right[found])
    a, b, group = a[found], b[found], group[found]
    if not len(a):
        return {"n_pairs": 0}
    brighter = np.where(old_common["r0"][a] <= old_common["r0"][b], a, b)
    fainter = np.where(brighter == a, b, a)
    old_delta = old_common["mu_median"][fainter] - old_common["mu_median"][brighter]
    new_delta = new_common["mu_median"][fainter] - new_common["mu_median"][brighter]
    change = new_delta - old_delta
    return {"n_pairs": len(a), "components": len(np.unique(group)),
            "old_fainter_minus_brighter": robust_summary(old_delta),
            "wise20_fainter_minus_brighter": robust_summary(new_delta),
            "wise20_minus_old_pair_delta": robust_summary(change),
            "wise20_minus_old_component_bootstrap95": component_median_ci(
                change, group)}


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen preparation changed")
    lock = json.loads(LOCK.read_text())
    if lock["prepared_sha256"] != sha(PREPARED):
        raise ValueError("Policy lock differs from frozen preparation")
    prepared = json.loads(PREPARED.read_text())
    route_path = OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen route changed")
    route = joblib.load(route_path)
    forests = {}
    for tier in FEATURES:
        path = OUT / f"{tier}.joblib"
        if sha(path) != prepared["tier_training"][tier]["model_sha256"]:
            raise ValueError(f"Frozen forest changed: {tier}")
        forests[tier] = joblib.load(path)
    refs = load_reference_catalogs("data/reference")
    old_rows, new_rows, fields = [], [], []
    for config in FINAL_CONFIGS:
        table, field = member_rows(config, refs)
        if table is None:
            fields.append(field)
            continue
        routed = route_rows(table, route)
        old_level = np.where(table["grz_clean"], routed["tier"], 0)
        new_level = choose_policy(table, routed, lock["chosen_policy"])
        old_rows.append(prediction_records(table, old_level, forests))
        new_rows.append(prediction_records(table, new_level, forests))
        field["old_selected"] = int(np.sum(old_level > 0))
        field["wise20_selected"] = int(np.sum(new_level > 0))
        field["old_tiers"] = {tier: int(np.sum(old_level == number))
                              for tier, number in LEVEL.items()}
        field["wise20_tiers"] = {tier: int(np.sum(new_level == number))
                                 for tier, number in LEVEL.items()}
        fields.append(field)
    old, new = np.concatenate(old_rows), np.concatenate(new_rows)
    old_pairs = evaluate_pairs(old, PAIR_CATALOG)
    new_pairs = evaluate_pairs(new, PAIR_CATALOG)
    result = {"prepared_sha256": sha(PREPARED), "policy_lock_sha256": sha(LOCK),
              "compute_host": socket.gethostname(), "relative_distance_only": True,
              "fields": fields, "old_clean_route": {
                  "binaries": old_pairs, "clusters": evaluate_clusters(old, refs)},
              "wise20_route": {"binaries": new_pairs,
                               "clusters": evaluate_clusters(new, refs)},
              "paired_same_binary_systems": paired_binary_change(old, new,
                                                                   PAIR_CATALOG)}
    write_json(RESULT, result)


if __name__ == "__main__":
    main()

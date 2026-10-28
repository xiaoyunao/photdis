#!/usr/bin/env python3
"""Relative-distance validation on reserved wide binaries and cluster members."""
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

from photdis.conditional import group_labels
from photdis.holdouts import load_reference_catalogs
from photdis.route import FEATURES, LEVEL, apply_route_offsets, feature_matrix, route_rows
from photdis.stage1 import require_compute, write_json
from cold_v4 import FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED, sha

RESULT = Path("data/cold_v4_systems/relative_validation.json")
READ = ["source_id", "g0", "r0", "rz0", "r", "gr0", "zw10", "w1w20",
        "grz_base", "grz_clean", "grz_w1", "grz_w1_w2"]


def member_rows(config_path, references):
    task = json.loads(Path(config_path).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    with fitsio.FITS(str(path)) as file:
        ids = file[1].read(columns=["source_id"])["source_id"]
        external = np.isin(ids, references.binary_source_id) | np.isin(ids, references.member_source_id)
        row_index = np.flatnonzero(external)
        if not len(row_index):
            return None, {"field": task["name"], "reference_rows": 0}
        columns = sorted(set(READ + ["sigma_g", "sigma_r", "sigma_z",
                                     "sigma_w1", "sigma_w2"]))
        raw = file[1].read(columns=columns, rows=row_index)
    unique, inverse, count = np.unique(raw["source_id"], return_inverse=True,
                                       return_counts=True)
    unambiguous = count[inverse] == 1
    raw = raw[unambiguous]
    return {name: raw[name] for name in raw.dtype.names}, {
        "field": task["name"], "reference_rows": len(row_index),
        "unique_Gaia_rows": len(raw),
        "duplicate_identity_rows_excluded": int(np.sum(~unambiguous)),
    }


def robust_summary(values):
    x = np.asarray(values, float)
    if not len(x):
        return {"n": 0}
    center = float(np.median(x))
    return {"n": len(x), "median": center,
            "nmad": float(1.4826 * np.median(np.abs(x - center))),
            "median_abs": float(np.median(np.abs(x))),
            "abs_gt_0p5_fraction": float(np.mean(np.abs(x) > .5))}


def evaluate_pairs(records, pair_catalog):
    ids = records["source_id"]
    order = np.argsort(ids)
    ids = ids[order]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Gaia reference identity repeats across fields")
    result = {"pairs_in_clean_routed_sample": 0, "components": 0,
              "fainter_minus_brighter_mu": {}, "redder_minus_bluer_mu": {},
              "color_difference_bins": [], "r_difference_bins": []}
    with np.load(pair_catalog, allow_pickle=False) as pairs:
        left = np.asarray(pairs["source_id1"], np.int64)
        right = np.asarray(pairs["source_id2"], np.int64)
        groups = np.asarray(pairs["group"], np.int64)
    a = np.searchsorted(ids, left)
    b = np.searchsorted(ids, right)
    found = (a < len(ids)) & (b < len(ids))
    found[found] &= (ids[a[found]] == left[found]) & (ids[b[found]] == right[found])
    if not found.any():
        return result
    i, j = order[a[found]], order[b[found]]
    pair_group = groups[found]
    mu = records["mu_median"]
    brighter = np.where(records["r0"][i] <= records["r0"][j], i, j)
    fainter = np.where(brighter == i, j, i)
    bluer = np.where(records["rz0"][i] <= records["rz0"][j], i, j)
    redder = np.where(bluer == i, j, i)
    delta_faint = mu[fainter] - mu[brighter]
    delta_red = mu[redder] - mu[bluer]
    color_diff = records["rz0"][redder] - records["rz0"][bluer]
    r_diff = records["r0"][fainter] - records["r0"][brighter]
    result["pairs_in_clean_routed_sample"] = len(i)
    result["components"] = len(np.unique(pair_group))
    result["fainter_minus_brighter_mu"] = robust_summary(delta_faint)
    result["redder_minus_bluer_mu"] = robust_summary(delta_red)
    for low, high in ((0, .2), (.2, .5), (.5, 5)):
        take = (color_diff >= low) & (color_diff < high)
        result["color_difference_bins"].append({"bin": [low, high],
                                                 "redder_minus_bluer_mu": robust_summary(delta_red[take])})
    for low, high in ((0, 1), (1, 2), (2, 10)):
        take = (r_diff >= low) & (r_diff < high)
        result["r_difference_bins"].append({"bin": [low, high],
                                             "fainter_minus_brighter_mu": robust_summary(delta_faint[take])})
    return result


def evaluate_clusters(records, references):
    ids = records["source_id"]
    group = np.zeros(len(ids), np.int64)
    position = np.searchsorted(references.member_source_id, ids)
    hit = position < len(references.member_source_id)
    hit[hit] &= references.member_source_id[position[hit]] == ids[hit]
    group[hit] = references.member_group[position[hit]]
    within = []
    per_group = []
    for number in np.unique(group[group > 0]):
        take = group == number
        if take.sum() < 5:
            continue
        mu = records["mu_median"][take]
        centered = mu - np.median(mu)
        within.extend(centered)
        per_group.append({"reference_group": int(number),
                          "n_clean_routed_members": int(take.sum()),
                          "relative_mu": robust_summary(centered)})
    return {"groups_with_at_least_5_members": len(per_group),
            "pooled_member_rows": len(within),
            "pooled_within_group_mu": robust_summary(within),
            "groups": per_group}


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Cold-v4 preparation changed")
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
    references = load_reference_catalogs("data/reference")
    arrays, fields = [], []
    edges = np.asarray(prepared["route_groups"]["color_edges"], float)
    r_edges = prepared["route_groups"]["r_edges_by_color"]
    for config in FINAL_CONFIGS:
        table, field = member_rows(config, references)
        if table is None:
            fields.append(field)
            continue
        routed = route_rows(table, route)
        selected = routed["selected"] & table["grz_clean"]
        pred = np.full((len(selected), 5), np.nan)
        for tier, level in LEVEL.items():
            take = selected & (routed["tier"] == level)
            if take.any():
                pred[take] = forests[tier].predict(feature_matrix(table, tier)[take])
        groups = group_labels(table["rz0"][selected], table["r"][selected],
                              edges, r_edges)
        adjusted = apply_route_offsets(pred[selected], groups,
                                       prepared["calibration_offsets"])
        output = np.empty(int(selected.sum()), dtype=[("source_id", "i8"),
                                                       ("r0", "f8"), ("rz0", "f8"),
                                                       ("mu_median", "f8")])
        for key in ("source_id", "r0", "rz0"):
            output[key] = table[key][selected]
        output["mu_median"] = table["g0"][selected] - adjusted[:, 2]
        arrays.append(output)
        field["clean_routed_reference_rows"] = len(output)
        field["tiers"] = {tier: int(np.sum(selected & (routed["tier"] == level)))
                          for tier, level in LEVEL.items()}
        fields.append(field)
    if not arrays:
        raise ValueError("No held-out binary or cluster members in selected fields")
    records = np.concatenate(arrays)
    result = {"prepared_sha256": sha(PREPARED), "compute_host": socket.gethostname(),
              "fields": fields, "clean_routed_reference_rows": len(records),
              "relative_distance_only": True,
              "binary_catalog": "data/reference/wide_binary_pairs_v1.npz",
              "binaries": evaluate_pairs(records, "data/reference/wide_binary_pairs_v1.npz"),
              "clusters": evaluate_clusters(records, references)}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, result)


if __name__ == "__main__":
    main()

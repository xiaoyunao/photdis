#!/usr/bin/env python3
"""Outcome-blind matching diagnostic of registered native final fields."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import healpy as hp
import numpy as np
from scipy.spatial import cKDTree

from photdis.holdouts import load_reference_catalogs
from photdis.route import LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import PREPARED, configured_field, sha
from cold_v4_policy_noise import frozen_models
from cold_v4_native_transfer import EXTRA, LOCK, REGISTRATION

NORTH = Path("configs/cold_v4_native_transfer_north_final.json")
OUT = Path("data/cold_v4_native_transfer/final_match.json")
FEATURE_SETS = {
    "legacy_colors_dust": (("gr0", .2), ("rz0", .15), ("ag", .02)),
    "plus_parallax": (("gr0", .2), ("rz0", .15), ("ag", .02),
                      ("p_corr", 1.0)),
    "plus_gaia_color": (("gr0", .2), ("rz0", .15), ("ag", .02),
                        ("bp_rp", .4)),
    "plus_parallax_gaia_color_r": (("gr0", .2), ("rz0", .15),
                                    ("ag", .02), ("p_corr", 1.0),
                                    ("bp_rp", .4), ("r0", .8)),
}


def selected_rows(config, references, route, forests, policy):
    table, cold, _, metadata = configured_field(config, references)
    with fitsio.FITS(metadata["path"]) as file:
        anchor_rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        extras = file[1].read(columns=list(EXTRA), rows=anchor_rows)
    if not np.array_equal(extras["source_id"], table["source_id"]):
        raise ValueError("Anchor order changed")
    table.update({key: extras[key] for key in EXTRA if key != "source_id"})
    level = choose_policy(table, route_rows(table, route), policy)
    use = cold & (level > 0)
    prediction = np.full(len(use), np.nan)
    for tier, number in LEVEL.items():
        mask = use & (level == number)
        if mask.any():
            prediction[mask] = np.asarray(table["g0"][mask], float) - \
                forests[tier].predict(feature_matrix(table, tier)[mask])[:, 2]
    p = np.asarray(table["p_corr"][use], float)
    residual = prediction[use] - (10 - 5 * np.log10(p))
    if not np.isfinite(residual).all():
        raise ValueError("Nonfinite residual")
    result = {
        "source_id": np.asarray(table["source_id"][use], np.int64),
        "residual": residual,
        "ra": np.asarray(table["ra"][use], float),
        "dec": np.asarray(table["dec"][use], float),
        "gr0": np.asarray(table["gr0"][use], float),
        "rz0": np.asarray(table["rz0"][use], float),
        "ag": -2.5 * np.log10(np.asarray(table["mw_transmission_g"][use], float)),
        "p_corr": p,
        "bp_rp": np.asarray(table["bp_rp_checked"][use], float),
        "r0": np.asarray(table["r0"][use], float),
    }
    return metadata["name"], result


def match_rows(south, north, features, subset, seed, radius=2.0):
    valid = []
    matrices = []
    for rows in (south, north):
        matrix = np.column_stack([np.asarray(rows[key], float) / scale
                                  for key, scale in features])
        mask = np.isfinite(matrix).all(axis=1) & np.isfinite(rows["residual"])
        if subset == "parallax_ge_2_mas":
            mask &= rows["p_corr"] >= 2
        elif subset == "parallax_1_to_2_mas":
            mask &= (rows["p_corr"] >= 1) & (rows["p_corr"] < 2)
        valid.append(np.flatnonzero(mask))
        matrices.append(matrix[mask])
    south_matrix, north_matrix = matrices
    if not len(south_matrix) or not len(north_matrix):
        return {"n": 0}
    tree = cKDTree(south_matrix)
    distance, candidates = tree.query(north_matrix, k=min(8, len(south_matrix)))
    distance = np.atleast_2d(distance).reshape(len(north_matrix), -1)
    candidates = np.atleast_2d(candidates).reshape(len(north_matrix), -1)
    possibilities = [(float(distance[i, j]), i, int(candidates[i, j]))
                     for i in range(len(north_matrix))
                     for j in range(distance.shape[1])
                     if distance[i, j] <= radius]
    possibilities.sort()
    used_north, used_south, pairs = set(), set(), []
    for d, i, j in possibilities:
        if i not in used_north and j not in used_south:
            used_north.add(i)
            used_south.add(j)
            pairs.append((i, j, d))
    if not pairs:
        return {"n": 0}
    ni = valid[1][np.array([p[0] for p in pairs])]
    si = valid[0][np.array([p[1] for p in pairs])]
    diff = north["residual"][ni] - south["residual"][si]
    rng = np.random.default_rng(seed)
    pair_boot = [np.median(diff[rng.integers(len(diff), size=len(diff))])
                 for _ in range(400)]
    blocks = hp.ang2pix(64, north["ra"][ni], north["dec"][ni],
                        nest=True, lonlat=True)
    groups = [np.flatnonzero(blocks == block) for block in np.unique(blocks)]
    block_boot = [np.median(diff[np.concatenate([groups[j] for j in
                  rng.integers(len(groups), size=len(groups))])])
                  for _ in range(400)] if len(groups) >= 5 else []
    return {"n": len(diff), "north_minus_south_median": float(np.median(diff)),
            "pair_bootstrap95": np.quantile(pair_boot, [.025, .975]).tolist(),
            "north_sky_block_bootstrap95": np.quantile(block_boot, [.025, .975]).tolist()
            if block_boot else None,
            "n_north_nside64_blocks": len(groups),
            "median_matching_radius": float(np.median([p[2] for p in pairs])),
            "p95_matching_radius": float(np.quantile([p[2] for p in pairs], .95)),
            "median_absolute_feature_difference": {
                key: float(np.median(abs(north[key][ni] - south[key][si])))
                for key, _ in features}}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    original = json.loads(REGISTRATION.read_text())
    registered_north = json.loads(NORTH.read_text())
    if (registered_north["parent_registration_sha256"] != sha(REGISTRATION)
            or original["prepared_sha256"] != sha(PREPARED)
            or original["policy_lock_sha256"] != sha(LOCK)):
        raise ValueError("Registration or frozen model/policy lock changed")
    policy = json.loads(LOCK.read_text())["chosen_policy"]
    route, forests = frozen_models()
    references = load_reference_catalogs("data/reference")
    configs = (original["independent_final_fields"] +
               registered_north["independent_final_fields"])
    fields = dict(selected_rows(item["field_config"], references,
                                route, forests, policy) for item in configs)
    north_name = registered_north["independent_final_fields"][0]["name"]
    north = fields[north_name]
    results = {}
    for index, item in enumerate(original["independent_final_fields"]):
        south_name = item["name"]
        south = fields[south_name]
        results[south_name] = {
            subset: {name: match_rows(south, north, features, subset,
                                      20261130 + index * 100 + j * 10 + k)
                     for k, (name, features) in enumerate(FEATURE_SETS.items())}
            for j, subset in enumerate(("all", "parallax_1_to_2_mas",
                                        "parallax_ge_2_mas"))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, {"method": "outcome-blind greedy one-to-one nearest-neighbor, "
                          "8 candidates, standardized Euclidean radius <=2; "
                          "diagnostic final-field comparison only",
                     "compute_host": socket.gethostname(),
                     "south_registration_sha256": sha(REGISTRATION),
                     "north_registration_sha256": sha(NORTH),
                     "policy_lock_sha256": sha(LOCK),
                     "selected_counts": {name: len(rows["residual"])
                                         for name, rows in fields.items()},
                     "feature_sets": {name: [list(item) for item in features]
                                      for name, features in FEATURE_SETS.items()},
                     "matches": results})


if __name__ == "__main__":
    main()

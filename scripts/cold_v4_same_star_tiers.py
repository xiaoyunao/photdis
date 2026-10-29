#!/usr/bin/env python3
"""Force frozen A/B/C on shared native stars and match complete inputs."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np
from scipy.spatial import cKDTree

from photdis.followup import identity_hash
from photdis.holdouts import load_reference_catalogs
from photdis.route import FEATURES, LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import PREPARED, configured_field, sha
from cold_v4_native_transfer import LOCK, REGISTRATION, robust
from cold_v4_policy_noise import frozen_models

NORTH = Path("configs/cold_v4_native_transfer_north_final.json")
OUT = Path("data/cold_v4_native_transfer/same_star_tiers.json")
EXTRA = ("source_id", "r", "r0", "grz_clean", "snr_w1", "snr_w2",
         "sigma_g", "sigma_r", "sigma_z", "sigma_w1", "sigma_w2")
COLOR_CALIPER = .02
LOG_ERROR_CALIPER = .5


def selected_field(item, references, route, forests, policy, published):
    table, cold, mg, meta = configured_field(item["field_config"], references)
    with fitsio.FITS(meta["path"]) as file:
        anchor_rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        extra = file[1].read(columns=list(EXTRA), rows=anchor_rows)
    if not np.array_equal(extra["source_id"], table["source_id"]):
        raise ValueError("Configured anchor order changed")
    table.update({name: extra[name] for name in EXTRA if name != "source_id"})
    routed = route_rows(table, route)
    level = choose_policy(table, routed, policy)
    baseline = cold & (level > 0)
    if identity_hash(table["source_id"][baseline]) != published["selected_source_id_sha256"]:
        raise ValueError("Frozen native selected-source identity changed")
    all_supported = baseline & table["grz_clean"] & (table["snr_w1"] >= 20) & (
        table["snr_w2"] >= 20)
    for tier in FEATURES:
        all_supported &= routed["support_by_tier"][tier]
    if np.any(all_supported & (level != 3)):
        raise ValueError("Shared three-tier sample did not select C")
    mu_ref = 10 - 5 * np.log10(table["p_corr"])
    predicted_mg = {}
    by_tier = {}
    for tier in FEATURES:
        mg_pred = np.full(len(cold), np.nan)
        use = all_supported | (baseline & (level == LEVEL[tier]))
        if use.any():
            mg_pred[use] = forests[tier].predict(feature_matrix(table, tier)[use])[:, 2]
        predicted_mg[tier] = mg_pred
        residual = table["g0"][all_supported] - mg_pred[all_supported] - mu_ref[all_supported]
        by_tier[tier] = {"gaia_residual": robust(residual),
                         "absolute_residual_gt_0p5": int(np.sum(abs(residual) > .5))}
    a = table["g0"][all_supported] - predicted_mg["A_grz"][all_supported]
    b = table["g0"][all_supported] - predicted_mg["B_grzW1"][all_supported]
    c = table["g0"][all_supported] - predicted_mg["C_grzW1W2"][all_supported]
    summary = {"field": meta["name"], "frozen_selected_count": int(baseline.sum()),
               "shared_three_tier_count": int(all_supported.sum()),
               "shared_fraction_of_frozen_selected": float(all_supported.sum()/baseline.sum()),
               "shared_source_id_sha256": identity_hash(table["source_id"][all_supported]),
               "by_tier": by_tier,
               "same_star_differences": {"mu_A_minus_C": robust(a-c),
                                         "mu_B_minus_C": robust(b-c),
                                         "mu_A_minus_B": robust(a-b)}}
    selected = np.flatnonzero(baseline)
    rows = {name: np.asarray(values)[selected] for name, values in table.items()
            if name in set(EXTRA) | set(FEATURES["C_grzW1W2"]) | {"g0", "p_corr", "ra", "dec"}}
    rows["level"] = level[selected]
    rows["mg_gaia"] = np.asarray(mg, dtype=float)[selected]
    rows["g0"] = np.asarray(rows["g0"], dtype=float)
    rows["mg_model"] = np.full(len(selected), np.nan)
    for tier, number in LEVEL.items():
        take = rows["level"] == number
        rows["mg_model"][take] = predicted_mg[tier][selected[take]]
    return meta["name"], summary, rows


def match_complete(south, north, tier):
    """Unique greedy pairs with absolute color and relative-error calipers."""
    level = LEVEL[tier]
    features = FEATURES[tier]
    s = np.flatnonzero(south["level"] == level)
    n = np.flatnonzero(north["level"] == level)
    if not len(s) or not len(n):
        return {"n": 0}
    sx = np.column_stack([south[key][s] for key in features])
    nx = np.column_stack([north[key][n] for key in features])
    tree = cKDTree(sx / COLOR_CALIPER)
    dist, candidates = tree.query(nx / COLOR_CALIPER, k=min(8, len(s)),
                                  distance_upper_bound=np.sqrt(len(features)))
    dist = np.asarray(dist).reshape(len(n), -1)
    candidates = np.asarray(candidates).reshape(len(n), -1)
    bands = ("g", "r", "z") + (("w1",) if level >= 2 else ()) + (("w2",) if level == 3 else ())
    pair_candidates = []
    for i in range(len(n)):
        for j in range(dist.shape[1]):
            k = int(candidates[i, j])
            if k >= len(s) or not np.isfinite(dist[i, j]):
                continue
            if np.max(abs(nx[i] - sx[k])) > COLOR_CALIPER:
                continue
            a = np.array([north[f"sigma_{band}"][n[i]] for band in bands])
            b = np.array([south[f"sigma_{band}"][s[k]] for band in bands])
            if (np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or
                    np.any(a <= 0) or np.any(b <= 0) or
                    np.max(abs(np.log(a/b))) > LOG_ERROR_CALIPER):
                continue
            pair_candidates.append((float(dist[i, j]), i, k))
    pair_candidates.sort()
    used_n, used_s, pairs = set(), set(), []
    for _, i, j in pair_candidates:
        if i not in used_n and j not in used_s:
            used_n.add(i); used_s.add(j); pairs.append((n[i], s[j]))
    if not pairs:
        return {"n": 0}
    ni = np.array([i for i, _ in pairs]); si = np.array([j for _, j in pairs])
    label = north["mg_gaia"][ni] - south["mg_gaia"][si]
    model = north["mg_model"][ni] - south["mg_model"][si]
    residual = label - model
    if not np.allclose(residual, (north["g0"][ni] - north["mg_model"][ni] -
                                  (north["g0"][ni] - north["mg_gaia"][ni])) -
                                  (south["g0"][si] - south["mg_model"][si] -
                                   (south["g0"][si] - south["mg_gaia"][si]))):
        raise ValueError("Pair residual decomposition failed")
    strata = {}
    for label_name, mask in (("both_parallax_ge_2", (north["p_corr"][ni] >= 2) &
                              (south["p_corr"][si] >= 2)),
                             ("both_r0_lt_17", (north["r0"][ni] < 17) &
                              (south["r0"][si] < 17)),
                             ("both_r0_ge_17", (north["r0"][ni] >= 17) &
                              (south["r0"][si] >= 17))):
        strata[label_name] = {"residual": robust(residual[mask]),
                              "label": robust(label[mask]), "model": robust(model[mask])}
    return {"n": len(pairs), "tier": tier, "color_caliper_mag": COLOR_CALIPER,
            "absolute_log_sigma_ratio_caliper": LOG_ERROR_CALIPER,
            "north_minus_south_residual": robust(residual),
            "north_minus_south_gaia_mg": robust(label),
            "north_minus_south_model_mg": robust(model),
            "median_absolute_color_difference": {key: float(np.median(abs(
                north[key][ni] - south[key][si]))) for key in features},
            "strata": strata}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    registration = json.loads(REGISTRATION.read_text())
    north = json.loads(NORTH.read_text())
    if (registration["prepared_sha256"] != sha(PREPARED) or
            north["parent_registration_sha256"] != sha(REGISTRATION) or
            registration["policy_lock_sha256"] != sha(LOCK)):
        raise ValueError("Frozen field or model registration changed")
    published = {x["field"]: x for path in (
        "manifests/cold_v4_native_transfer_final.json",
        "manifests/cold_v4_native_transfer_north_final.json") for x in
        json.loads(Path(path).read_text())["fields"]}
    route, forests = frozen_models()
    references = load_reference_catalogs("data/reference")
    fields = {}
    rows = {}
    for item in registration["independent_final_fields"] + north["independent_final_fields"]:
        name, fields[item["name"]], rows[item["name"]] = selected_field(
            item, references, route, forests,
            json.loads(LOCK.read_text())["chosen_policy"], published[item["name"]])
        print(name, fields[name]["shared_three_tier_count"], flush=True)
    north_name = north["independent_final_fields"][0]["name"]
    matching = {item["name"]: {tier: match_complete(rows[item["name"]],
        rows[north_name], tier) for tier in LEVEL}
        for item in registration["independent_final_fields"]}
    write_json(OUT, {"method": "frozen v4; exact native source identities; common supported A/B/C; outcome-blind complete-input matching on already scored fields",
                     "compute_host": socket.gethostname(),
                     "registration_sha256": sha(REGISTRATION),
                     "north_registration_sha256": sha(NORTH),
                     "fields": fields, "north_south_complete_input_matches": matching})


if __name__ == "__main__":
    main()

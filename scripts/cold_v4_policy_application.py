#!/usr/bin/env python3
"""Count actual DR11 candidates under the development-locked clean WISE route."""
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
from photdis.noise import interval_diagnostics, transformed_distance_quantiles
from photdis.route import FEATURES, LAYER, LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import FINAL_CONFIGS, OUT, PREPARED, PREPARED_TRACKED, configured_field, sha

LOCK = Path("configs/cold_v4_deep_policy.json")
RESULT = Path("data/cold_v4_deep/application_policy.json")
COLS = sorted(set([*FEATURES["C_grzW1W2"], *LAYER.values(),
                   "grz_clean", "grz_high_quality", "grz_w1_high_quality",
                   "full_high_quality", "snr_w1", "snr_w2", "r",
                   "sigma_g", "sigma_r", "sigma_z", "sigma_w1", "sigma_w2"]))
R_BINS = ((14, 18), (18, 20), (20, 21), (21, 22), (22, 26))


def field_result(config_path, route, policy):
    task = json.loads(Path(config_path).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    raw = fitsio.read(path, columns=COLS)
    table = {name: raw[name] for name in raw.dtype.names}
    baseline = route_rows(table, route)
    chosen = choose_policy(table, baseline, policy)
    clean = np.asarray(table["grz_clean"], bool)
    selected = chosen > 0
    old = baseline["tier"]
    result = {"field": task["name"], "stored_rows": len(raw),
              "grz_base": int(table["grz_base"].sum()),
              "baseline_selected": int(baseline["selected"].sum()),
              "baseline_selected_and_clean": int(np.sum(baseline["selected"] & clean)),
              "locked_policy_selected": int(selected.sum()),
              "locked_policy_selected_nonclean": int(np.sum(selected & ~clean)),
              "transition_counts_old_to_new": {}, "new_tiers": {}, "by_r": []}
    for old_level in range(4):
        result["transition_counts_old_to_new"][str(old_level)] = {
            str(new_level): int(np.sum((old == old_level) & (chosen == new_level)))
            for new_level in range(4)}
    quality = {"A_grz": "grz_high_quality", "B_grzW1": "grz_w1_high_quality",
               "C_grzW1W2": "full_high_quality"}
    for tier, level in LEVEL.items():
        take = chosen == level
        result["new_tiers"][tier] = {"selected": int(take.sum()),
                                     "used_tier_high_quality": int(np.sum(take & table[quality[tier]])),
                                     "optical_high_quality": int(np.sum(take & table["grz_high_quality"]))}
    for lo, hi in R_BINS:
        within = (table["r"] >= lo) & (table["r"] < hi)
        result["by_r"].append({"r_bin": [lo, hi],
                               "grz_base": int(np.sum(within & table["grz_base"])),
                               "baseline_selected": int(np.sum(within & baseline["selected"])),
                               "locked_policy_selected": int(np.sum(within & selected)),
                               "new_tiers": {tier: int(np.sum(within & (chosen == level)))
                                             for tier, level in LEVEL.items()}})
    if result["locked_policy_selected_nonclean"]:
        raise ValueError("Locked clean route selected nonclean photometry")
    if sum(v["selected"] for v in result["new_tiers"].values()) != result["locked_policy_selected"]:
        raise ValueError("Tier count mismatch")
    if result["locked_policy_selected"] > result["baseline_selected_and_clean"]:
        raise ValueError("Locked route added candidates outside clean baseline")
    return result


def strict_gaia_result(config_path, route, policy, forests, references):
    """Direct-model check of the locked routing on independent strict anchors."""
    table, cold, _, metadata = configured_field(config_path, references)
    with fitsio.FITS(metadata["path"]) as file:
        rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        extra = file[1].read(columns=["source_id", "grz_clean", "grz_high_quality",
                                      "grz_w1_high_quality", "full_high_quality",
                                      "snr_w1", "snr_w2"], rows=rows)
    if not np.array_equal(extra["source_id"], table["source_id"]):
        raise ValueError("Gaia sidecar differs from anchor order")
    table.update({name: extra[name] for name in extra.dtype.names if name != "source_id"})
    baseline = route_rows(table, route)
    chosen = choose_policy(table, baseline, policy)
    take = cold & (chosen > 0)
    predicted = np.full((len(take), 5), np.nan)
    for tier, level in LEVEL.items():
        use = take & (chosen == level)
        if use.any():
            predicted[use] = forests[tier].predict(feature_matrix(table, tier)[use])
    if not np.isfinite(predicted[take]).all():
        raise ValueError("Nonfinite strict-Gaia locked-route prediction")
    truth = 10 - 5 * np.log10(np.asarray(table["p_corr"][take], float))
    modulus = transformed_distance_quantiles(np.asarray(table["g0"][take], float),
                                              predicted[take])[0]
    selected_tier = chosen[take]
    result = {"strict_cold_anchor_rows": int(cold.sum()),
              "locked_selected": int(take.sum()),
              "direct_no_new_CQR": interval_diagnostics(truth, modulus),
              "by_tier": {}}
    for tier, level in LEVEL.items():
        use = selected_tier == level
        result["by_tier"][tier] = (interval_diagnostics(truth[use], modulus[use])
                                   if use.any() else {"n": 0})
    return result


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen preparation changed")
    lock = json.loads(LOCK.read_text())
    if lock["prepared_sha256"] != sha(PREPARED):
        raise ValueError("Deep policy does not match v4 preparation")
    route_path = OUT / "route.joblib"
    prepared = json.loads(PREPARED.read_text())
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
    report = {"prepared_sha256": sha(PREPARED),
              "policy_lock_sha256": sha(LOCK),
              "compute_host": socket.gethostname(),
              "photometric_candidates_no_truth_labels": True,
              "fields": {}}
    for config in FINAL_CONFIGS:
        result = field_result(config, route, lock["chosen_policy"])
        result["strict_Gaia_locked_route"] = strict_gaia_result(
            config, route, lock["chosen_policy"], forests, references)
        report["fields"][result["field"]] = result
        print(json.dumps({"field": result["field"],
                          "baseline": result["baseline_selected"],
                          "locked": result["locked_policy_selected"]}), flush=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, report)


if __name__ == "__main__":
    main()

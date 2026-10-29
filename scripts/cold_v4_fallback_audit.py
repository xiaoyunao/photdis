#!/usr/bin/env python3
"""Development-field audit of reliable infrared support conflicts."""
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
from photdis.holdouts import load_reference_catalogs
from photdis.route import LEVEL, feature_matrix, route_rows
from photdis.route_audit import higher_tier_reasons
from photdis.route_policy import choose_policy
from photdis.stage1 import legacy_ids, require_compute, write_json
from cold_v4 import PREPARED, PREPARED_TRACKED, OUT, configured_field, sha
from cold_v4_type_audit import COLS, class_masks, unique_spectra

LOCK = Path("configs/cold_v4_deep_policy.json")
REGISTRATION = Path("configs/cold_v4_native_transfer_fields.json")
OUTPUT = Path("data/cold_v4_fallback/development.json")
DEVELOPMENT = ("configs/field_dev_cal_ra205_215.json",)
RAW = ("RELEASE", "BRICKID", "OBJID", "WISEMASK_W1", "WISEMASK_W2",
       "FRACFLUX_W1", "FRACFLUX_W2", "NOBS_W1", "NOBS_W2")
READ = sorted(set(COLS + ["anchor", "grz_base", "valid_w1", "valid_w2",
                          "w1_clean", "w2_clean", "snr_w1", "snr_w2",
                          "ebv", "g0", "r0", "p_corr", "gaia_bp_rp"]))


def support_ratio(table, route, tier, index):
    model = route["tiers"][tier]
    colors = feature_matrix(table, tier)[index]
    result = np.full(len(index), np.nan)
    finite = np.isfinite(colors).all(axis=1)
    if finite.any():
        result[finite] = model["tree"].query(colors[finite] / model["scale"],
            k=route["neighbors"])[0][:, -1] / model["radius"]
    return result


def boundary_stability(table, route, chosen, conflict, seed, limit=5000, draws=64):
    ids = np.flatnonzero(conflict)
    if len(ids) > limit:
        ids = np.sort(np.random.default_rng(seed).choice(ids, limit, replace=False))
    rng = np.random.default_rng(seed + 1)
    values = []
    for tier, level in (("B_grzW1", 1), ("C_grzW1W2", 2)):
        use = ids[chosen[ids] == level]
        if not len(use):
            continue
        bands = ("g", "r", "z", "w1") if level == 1 else ("g", "r", "z", "w1", "w2")
        sigma = np.column_stack([table[f"sigma_{b}"][use] for b in bands]).astype(float)
        if not np.isfinite(sigma).all():
            raise ValueError("Conflicted good-quality WISE colors have nonfinite band errors")
        baseline = support_ratio(table, route, tier, use)
        model = route["tiers"][tier]
        for start in range(0, len(use), 200):
            stop = min(start + 200, len(use))
            magnitude = np.column_stack([table[f"{b}0"][use[start:stop]]
                                         for b in bands]).astype(float)
            perturbed = magnitude[:, None, :] + rng.normal(size=(stop-start, draws, len(bands))) * sigma[start:stop, None, :]
            colors = np.diff(-perturbed, axis=2).reshape(-1, len(bands)-1)
            radius = model["tree"].query(colors / model["scale"],
                k=route["neighbors"])[0][:, -1] / model["radius"]
            probability = np.mean(radius.reshape(stop-start, draws) > 1, axis=1)
            values.extend({"ls_id": int(table["ls_id"][use[j]]), "tier": tier,
                           "nominal_support_radius_ratio": float(baseline[j]),
                           "fraction_of_draws_unsupported": float(probability[j-start])}
                          for j in range(start, stop))
    return {"n_all_conflicts": int(conflict.sum()), "n_tested": len(values),
            "draws_per_source": draws, "seed": seed,
            "fraction_stable_conflict_ge_0p9": float(np.mean([
                v["fraction_of_draws_unsupported"] >= .9 for v in values])) if values else None,
            "boundary_ratio_1_to_1p2": [v for v in values if
                1 <= v["nominal_support_radius_ratio"] <= 1.2][:100]}


def raw_wise_quality(task, requested):
    source = json.loads(Path("configs/stage1.json").read_text())["sweep_root"]
    found = {}
    for name in task["sweep_files"]:
        raw = fitsio.read(str(Path(source) / name), columns=list(RAW))
        ids = legacy_ids(raw)
        take = np.flatnonzero(np.isin(ids, requested))
        for i in take:
            key = int(ids[i])
            if key in found:
                raise ValueError(f"Duplicate raw Legacy ID {key}")
            found[key] = {"wisemask_w1": int(raw["WISEMASK_W1"][i]),
                          "wisemask_w2": int(raw["WISEMASK_W2"][i]),
                          "fracflux_w1": float(raw["FRACFLUX_W1"][i]),
                          "fracflux_w2": float(raw["FRACFLUX_W2"][i]),
                          "nobs_w1": int(raw["NOBS_W1"][i]),
                          "nobs_w2": int(raw["NOBS_W2"][i])}
    if len(found) != len(requested):
        raise ValueError(f"Raw WISE quality matched {len(found)}/{len(requested)} IDs")
    return found


def count(mask, selected, conflict):
    before = int(np.sum(mask & selected))
    held = int(np.sum(mask & selected & conflict))
    return {"labeled": int(mask.sum()), "v4_accepted": before,
            "conflict_flagged": held, "candidate_hold_accepted": before-held,
            "relative_recall_loss": held/before if before else None}


def field_result(config, route, policy, references, seed):
    task = json.loads(Path(config).read_text())
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    raw = fitsio.read(str(path), columns=READ)
    table = {name: raw[name] for name in raw.dtype.names}
    routed = route_rows(table, route)
    chosen = choose_policy(table, routed, policy)
    reasons = higher_tier_reasons(table, routed, chosen)
    selected = chosen > 0
    conflict = reasons["higher_support_conflict"]
    if np.any(conflict & ~selected):
        raise ValueError("Conflict flag applied to rejected source")
    unique, dedup = unique_spectra(table)
    labels = class_masks(table, unique & table["grz_base"])
    feh, valid = rvs_metallicity(table["desi_feh"], table["desi_teff"],
                                table["desi_logg"])
    poor = labels["cool_dwarf"] & valid & np.isfinite(table["desi_feh_err"]) & (
        table["desi_feh_err"] > 0) & (feh < -1.5)
    anchors, cold, _, metadata = configured_field(config, references)
    gaia_cold = np.isin(table["source_id"], anchors["source_id"][cold])
    if int(gaia_cold.sum()) != int(cold.sum()):
        raise ValueError("Exact Gaia cold-anchor identities did not join")
    requested = np.unique(table["ls_id"][conflict | (labels["cool_giant"] & selected)])
    raw_quality = raw_wise_quality(task, requested)
    label_summary = {name: count(mask, selected, conflict) for name, mask in {
        **labels, "calibrated_metal_poor_cool_dwarf": poor,
        "gaia_strict_cold_dwarf": gaia_cold,
        "unlabeled_clean_candidate": table["grz_clean"] & selected & ~unique,
    }.items()}
    per_tier = {}
    for tier, level in LEVEL.items():
        use = selected & (chosen == level)
        higher = "B" if level == 1 else "C" if level == 2 else None
        per_tier[tier] = {"selected": int(use.sum()),
            "higher_reason": {reason: int(np.sum(use & (reasons[higher] == reason)))
                              for reason in np.unique(reasons[higher])} if higher else {}}
    for name, mask in (("gr0_lt_1p1", table["gr0"] < 1.1),
                       ("gr0_1p1_to_1p6", (table["gr0"] >= 1.1) & (table["gr0"] < 1.6)),
                       ("gr0_ge_1p6", table["gr0"] >= 1.6),
                       ("rz0_ge_1", table["rz0"] >= 1)):
        label_summary[name] = count(gaia_cold & mask, selected, conflict)
    giant_objects = []
    for i in np.flatnonzero(labels["cool_giant"] & selected):
        key = int(table["ls_id"][i])
        giant_objects.append({"ls_id": key, "source_id": int(table["source_id"][i]),
            "chosen_tier": int(chosen[i]), "higher_reason_B": str(reasons["B"][i]),
            "higher_reason_C": str(reasons["C"][i]),
            "snr_w1": float(table["snr_w1"][i]), "snr_w2": float(table["snr_w2"][i]),
            "w1_clean": bool(table["w1_clean"][i]),
            "w2_clean": bool(table["w2_clean"][i]),
            "B_radius_ratio": float(support_ratio(table, route, "B_grzW1", np.array([i]))[0]),
            "C_radius_ratio": float(support_ratio(table, route, "C_grzW1W2", np.array([i]))[0]),
            **raw_quality[key]})
    candidate = selected & table["grz_clean"]
    return {"field": task["name"], "source_path": str(path),
            "deduplication": dedup, "baseline_selected_clean": int(candidate.sum()),
            "candidate_conflict_count": int(np.sum(candidate & conflict)),
            "candidate_conflict_fraction": float(np.mean(conflict[candidate])) if candidate.any() else None,
            "baseline_tier_counts_and_reasons": per_tier,
            "labels": label_summary, "accepted_cold_giant_raw_wise": giant_objects,
            "raw_wise_conflict_summary": {
                "n": len(raw_quality),
                "nonzero_wisemask_w1": sum(v["wisemask_w1"] != 0 for v in raw_quality.values()),
                "nonzero_wisemask_w2": sum(v["wisemask_w2"] != 0 for v in raw_quality.values()),
                "fracflux_w1_ge_1": sum(v["fracflux_w1"] >= 1 for v in raw_quality.values()),
                "fracflux_w2_ge_1": sum(v["fracflux_w2"] >= 1 for v in raw_quality.values())},
            "boundary_stability": boundary_stability(table, route, chosen, conflict, seed)}


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen v4 preparation changed")
    prepared = json.loads(PREPARED.read_text())
    route_path = OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen v4 support changed")
    policy = json.loads(LOCK.read_text())["chosen_policy"]
    route = joblib.load(route_path)
    references = load_reference_catalogs("data/reference")
    native = json.loads(REGISTRATION.read_text())["development_fields"]
    configs = [*DEVELOPMENT, *(item["field_config"] for item in native)]
    result = {"method": "development labels only; frozen v4 route plus diagnostic conflict marker",
              "prepared_sha256": sha(PREPARED), "policy_lock_sha256": sha(LOCK),
              "compute_host": socket.gethostname(), "fields": {}}
    for i, config in enumerate(configs):
        field = field_result(config, route, policy, references, 20261210+i)
        print(json.dumps({"field": field["field"], "selected": field["baseline_selected_clean"],
                          "conflicts": field["candidate_conflict_count"]}), flush=True)
        result["fields"][field.pop("field")] = field
    write_json(OUTPUT, result)


if __name__ == "__main__":
    main()

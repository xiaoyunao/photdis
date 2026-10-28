#!/usr/bin/env python3
"""Bounded dust-correction sensitivity of the frozen native final route."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.followup import identity_hash
from photdis.holdouts import load_reference_catalogs
from photdis.route import LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import PREPARED, configured_field, sha
from cold_v4_policy_noise import frozen_models
from cold_v4_native_transfer import EXTRA, LOCK, REGISTRATION, robust, spatial_bootstrap

NORTH = Path("configs/cold_v4_native_transfer_north_final.json")
OUT = Path("data/cold_v4_dust_sensitivity/final.json")
COEFF = {"g": 3.214, "r": 2.165, "z": 1.211, "w1": .184, "w2": .113}
FRACTIONS = (1.0, .75, .5, 0.0)


def shift_dereddening(table, ag, fraction):
    """Change the assumed fraction of DR11 foreground A while keeping flux fixed."""
    if not 0 <= fraction <= 1:
        raise ValueError("Extinction fraction must be within 0..1")
    ag = np.asarray(ag, float)
    added = {band: (1 - fraction) * ag * coefficient / COEFF["g"]
             for band, coefficient in COEFF.items()}
    shifted = dict(table)
    shifted["g0"] = np.asarray(table["g0"], float) + added["g"]
    shifted["r0"] = np.asarray(table["r0"], float) + added["r"]
    shifted["gr0"] = np.asarray(table["gr0"], float) + added["g"] - added["r"]
    shifted["rz0"] = np.asarray(table["rz0"], float) + added["r"] - added["z"]
    shifted["zw10"] = np.asarray(table["zw10"], float) + added["z"] - added["w1"]
    shifted["w1w20"] = np.asarray(table["w1w20"], float) + added["w1"] - added["w2"]
    return shifted


def score(table, cold, route, forests, policy):
    level = choose_policy(table, route_rows(table, route), policy)
    selected = cold & (level > 0)
    prediction = np.full(len(selected), np.nan)
    for tier, number in LEVEL.items():
        use = selected & (level == number)
        if use.any():
            prediction[use] = np.asarray(table["g0"][use], float) - \
                forests[tier].predict(feature_matrix(table, tier)[use])[:, 2]
    return level, selected, prediction


def field_result(item, references, route, forests, policy, expected, seed):
    table, cold, _, metadata = configured_field(item["field_config"], references)
    with fitsio.FITS(metadata["path"]) as file:
        anchor_rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        extras = file[1].read(columns=list(EXTRA), rows=anchor_rows)
    if not np.array_equal(extras["source_id"], table["source_id"]):
        raise ValueError("Anchor order differs from dust extras")
    table.update({name: extras[name] for name in EXTRA if name != "source_id"})
    ag = -2.5 * np.log10(np.asarray(table["mw_transmission_g"], float))
    ebv = np.asarray(table["ebv"], float)
    level_base, selected_base, prediction_base = score(
        table, cold, route, forests, policy)
    if identity_hash(table["source_id"][selected_base]) != expected["selected_source_id_sha256"]:
        raise ValueError("Frozen selected identity differs from final score")
    p = np.asarray(table["p_corr"], float)
    mu_ref = 10 - 5 * np.log10(p[cold])
    if not np.isfinite(mu_ref).all():
        raise ValueError("Cold-anchor reference distance is not finite")
    results = {}
    for fraction in FRACTIONS:
        phot = table if fraction == 1 else shift_dereddening(table, ag, fraction)
        level, selected, prediction = (level_base, selected_base, prediction_base) \
            if fraction == 1 else score(phot, cold, route, forests, policy)
        common = selected_base & selected
        delta = prediction[common] - prediction_base[common]
        true_mu = 10 - 5 * np.log10(p[common])
        near = common & (p >= 2)
        delta_near = prediction[near] - prediction_base[near]
        key = f"fraction_{fraction:g}"
        results[key] = {
            "selected_cold_count": int(selected.sum()),
            "common_with_baseline_count": int(common.sum()),
            "baseline_acceptance_fraction": float(common.sum() / max(selected_base.sum(), 1)),
            "new_from_cold_count": int((selected & ~selected_base).sum()),
            "tier_changes_on_common": int((level[common] != level_base[common]).sum()),
            "paired_prediction_shift_mag": robust(delta),
            "paired_prediction_shift_sky_block": spatial_bootstrap(
                delta, np.asarray(table["ra"][common], float),
                np.asarray(table["dec"][common], float), seed + int(fraction * 100)),
            "same_source_baseline_residual": robust(prediction_base[common] - true_mu),
            "same_source_assumed_residual": robust(prediction[common] - true_mu),
            "parallax_ge_2_mas": {
                "paired_prediction_shift_mag": robust(delta_near),
                "paired_prediction_shift_sky_block": spatial_bootstrap(
                    delta_near, np.asarray(table["ra"][near], float),
                    np.asarray(table["dec"][near], float), seed + 1000 + int(fraction * 100)),
                "same_source_baseline_residual": robust(prediction_base[near] -
                    (10 - 5 * np.log10(p[near]))),
                "same_source_assumed_residual": robust(prediction[near] -
                    (10 - 5 * np.log10(p[near]))),
            },
        }
    catalog_ag = ag[np.isfinite(ebv) & (ebv > 0)]
    expected_ag = COEFF["g"] * ebv[np.isfinite(ebv) & (ebv > 0)]
    return {"field": metadata["name"], "baseline_selected_count": int(selected_base.sum()),
            "median_ebv": float(np.median(ebv[selected_base])),
            "median_catalog_ag": float(np.median(ag[selected_base])),
            "max_abs_catalog_ag_minus_coefficient_ebv": float(np.max(abs(catalog_ag - expected_ag))),
            "fractions": results}


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    original = json.loads(REGISTRATION.read_text())
    north = json.loads(NORTH.read_text())
    if (north["parent_registration_sha256"] != sha(REGISTRATION)
            or original["prepared_sha256"] != sha(PREPARED)
            or original["policy_lock_sha256"] != sha(LOCK)):
        raise ValueError("Registration or frozen model/policy lock changed")
    south_score = json.loads(Path("manifests/cold_v4_native_transfer_final.json").read_text())
    north_score = json.loads(Path("manifests/cold_v4_native_transfer_north_final.json").read_text())
    expected = {field["field"]: field for field in
                south_score["fields"] + north_score["fields"]}
    route, forests = frozen_models()
    references = load_reference_catalogs("data/reference")
    policy = json.loads(LOCK.read_text())["chosen_policy"]
    configs = original["independent_final_fields"] + north["independent_final_fields"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, {"method": "freeze Gaia cold anchors and true parallax; scale "
                          "the DR11 full-column extinction subtracted from "
                          "all five bands, reroute and repredict on the same sources",
                     "interpretation": "sensitivity bound, not a 3D dust estimate or fitted correction",
                     "compute_host": socket.gethostname(),
                     "frozen_policy_sha256": sha(LOCK),
                     "model_prepared_sha256": sha(PREPARED),
                     "coefficients_per_ebv": COEFF,
                     "assumed_foreground_fractions": FRACTIONS,
                     "fields": [field_result(item, references, route, forests,
                                             policy, expected[item["name"]],
                                             20261220 + index * 100)
                                for index, item in enumerate(configs)]})


if __name__ == "__main__":
    main()

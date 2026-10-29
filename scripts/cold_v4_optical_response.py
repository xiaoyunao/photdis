#!/usr/bin/env python3
"""Fixed-tier per-band PS1 replacement and small photometric response audit."""
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
from cold_v4_dust_sensitivity import score
from cold_v4_native_transfer import LOCK, REGISTRATION, robust
from cold_v4_policy_noise import frozen_models
from cold_v4_ps1_relative_audit import (EXTRA, NORTH, CONFIG, match_ps1,
                                         ps1_catalog, substitute_grz)

OUT = Path("data/cold_v4_optical_response/diagnostic.json")
AMPLITUDES = (.005, .01, .02)


def fixed_tier_score(table, level, mask, forests):
    result = np.full(len(level), np.nan)
    for tier, number in LEVEL.items():
        take = mask & (level == number)
        if take.any():
            result[take] = table["g0"][take] - forests[tier].predict(
                feature_matrix(table, tier)[take])[:, 2]
    return result


def band_substitution(table, mask, transformed, bands):
    values = {band: (transformed[band] if band in bands else table[band])
              for band in "grz"}
    return substitute_grz(table, mask, values)[0]


def comparison(table, shifted, old_mu, new_mu, mask, p):
    ref = 10 - 5 * np.log10(p[mask])
    old = old_mu[mask] - ref
    new = new_mu[mask] - ref
    dg = shifted["g0"][mask] - table["g0"][mask]
    dmu = new_mu[mask] - old_mu[mask]
    dmodel = dg - dmu
    return {"n": int(mask.sum()), "baseline_residual": robust(old),
            "shifted_residual": robust(new),
            "baseline_tail_abs_gt_0p5": int(np.sum(abs(old) > .5)),
            "shifted_tail_abs_gt_0p5": int(np.sum(abs(new) > .5)),
            "paired_mu_shift": robust(dmu),
            "apparent_g0_shift": robust(dg),
            "predicted_Mg_shift": robust(dmodel)}


def response_strata(table, old_mu, new_mu, base, support_ratio):
    shift = new_mu - old_mu
    result = {}
    groups = {"gr0_lt_1p2": table["gr0"] < 1.2,
              "gr0_1p2_to_1p6": (table["gr0"] >= 1.2) & (table["gr0"] < 1.6),
              "gr0_ge_1p6": table["gr0"] >= 1.6,
              "support_ratio_lt_0p7": support_ratio < .7,
              "support_ratio_0p7_to_0p9": (support_ratio >= .7) & (support_ratio < .9),
              "support_ratio_ge_0p9": support_ratio >= .9}
    for name, mask in groups.items():
        use = base & mask
        result[name] = {"shift": robust(shift[use]),
                        "absolute_mu_shift_gt_0p05": int(np.sum(abs(shift[use]) > .05))}
    return result


def field_result(item, references, route, forests, policy, expected):
    table, cold, _, metadata = configured_field(item["field_config"], references)
    with fitsio.FITS(metadata["path"]) as file:
        anchor_rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        extras = file[1].read(columns=list(EXTRA), rows=anchor_rows)
    if not np.array_equal(extras["source_id"], table["source_id"]):
        raise ValueError("Anchor/PS1 extras identity changed")
    table.update({name: extras[name] for name in EXTRA if name != "source_id"})
    old_level, baseline, old_mu = score(table, cold, route, forests, policy)
    if identity_hash(table["source_id"][baseline]) != expected["selected_source_id_sha256"]:
        raise ValueError("Frozen native selected source hash changed")
    selected = np.flatnonzero(baseline)
    gaia = {name: table[name][selected] for name in EXTRA if name.startswith("gaia_")}
    catalog, inventory = ps1_catalog(item["bounds_deg"])
    quality_selected, transformed_selected, match = match_ps1(gaia, catalog)
    quality = np.zeros(len(baseline), bool)
    quality[selected] = quality_selected
    transformed = {band: np.asarray(table[band], float).copy() for band in "grz"}
    for band in "grz":
        transformed[band][selected] = transformed_selected[band]
    fixed = baseline & quality
    support_ratio = np.full(len(fixed), np.nan)
    for tier, number in LEVEL.items():
        use = fixed & (old_level == number)
        if use.any():
            model = route["tiers"][tier]
            support_ratio[use] = model["tree"].query(
                feature_matrix(table, tier)[use] / model["scale"],
                k=route["neighbors"])[0][:, -1] / model["radius"]
    p = np.asarray(table["p_corr"], float)
    result = {"field": metadata["name"], "frozen_selected": int(baseline.sum()),
              "ps1_quality_fixed_sample": int(fixed.sum()),
              "ps1_quality_source_id_sha256": identity_hash(table["source_id"][fixed]),
              "matching": match, "ps1_inventory": inventory,
              "baseline": comparison(table, table, old_mu, old_mu, fixed, p),
              "ps1_substitutions": {}, "finite_perturbations": {}}
    for bands in ("g", "r", "z", "grz"):
        shifted = band_substitution(table, fixed, transformed, bands)
        fixed_mu = fixed_tier_score(shifted, old_level, fixed, forests)
        shifted_route = route_rows(shifted, route)
        supported = np.zeros(len(fixed), bool)
        for tier, number in LEVEL.items():
            take = fixed & (old_level == number)
            supported[take] = shifted_route["support_by_tier"][tier][take]
        new_level, new_accepted, rerouted_mu = score(
            shifted, cold, route, forests, policy)
        common = fixed & new_accepted
        result["ps1_substitutions"][bands] = {
            "fixed_tier": comparison(table, shifted, old_mu, fixed_mu, fixed, p),
            "fixed_tier_response_strata": response_strata(
                table, old_mu, fixed_mu, fixed, support_ratio),
            "fixed_tier_still_supported": int(np.sum(fixed & supported)),
            "rerouted_common": comparison(table, shifted, old_mu, rerouted_mu, common, p),
            "rerouted_tier_changes": int(np.sum(common & (new_level != old_level))),
            "rerouted_rejections": int(np.sum(fixed & ~new_accepted))}
    for band in "grz":
        values = {}
        for size in AMPLITUDES:
            for sign in (-1, 1):
                delta = sign * size
                modified = dict(transformed)
                modified[band] = np.asarray(table[band], float) + delta
                shifted = band_substitution(table, fixed, modified, band)
                new_mu = fixed_tier_score(shifted, old_level, fixed, forests)
                values[f"{delta:+.3f}"] = {
                    "overall": comparison(table, shifted, old_mu, new_mu, fixed, p),
                    "response_strata": response_strata(
                        table, old_mu, new_mu, fixed, support_ratio)}
        result["finite_perturbations"][band] = values
    return result


def main():
    require_compute()
    if OUT.exists():
        raise FileExistsError(OUT)
    registration = json.loads(REGISTRATION.read_text())
    north = json.loads(NORTH.read_text())
    if (registration["prepared_sha256"] != sha(PREPARED) or
            north["parent_registration_sha256"] != sha(REGISTRATION) or
            registration["policy_lock_sha256"] != sha(LOCK)):
        raise ValueError("Frozen native registration changed")
    published = {x["field"]: x for path in (
        "manifests/cold_v4_native_transfer_final.json",
        "manifests/cold_v4_native_transfer_north_final.json") for x in
        json.loads(Path(path).read_text())["fields"]}
    route, forests = frozen_models()
    references = load_reference_catalogs("data/reference")
    fields = {}
    for item in registration["independent_final_fields"] + north["independent_final_fields"]:
        row = field_result(item, references, route, forests,
                           json.loads(LOCK.read_text())["chosen_policy"],
                           published[item["name"]])
        fields[row.pop("field")] = row
        print(item["name"], row["ps1_quality_fixed_sample"], flush=True)
    write_json(OUT, {"method": "fixed frozen tier per-band transformed PS1 replacement and symmetric finite diagnostic perturbations; rerouting separate",
                     "compute_host": socket.gethostname(),
                     "registration_sha256": sha(REGISTRATION),
                     "ps1_config_sha256": sha(CONFIG), "fields": fields})


if __name__ == "__main__":
    main()

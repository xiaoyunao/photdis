#!/usr/bin/env python3
"""Compare preregistered v4 route policies on one shared development noise draw."""
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

from photdis.followup import identity_hash
from photdis.holdouts import load_reference_catalogs
from photdis.noise import simulate_shared_flux
from photdis.photometry import derive_photometry
from photdis.route import FEATURES, LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from cold_v4 import OUT, PREPARED, PREPARED_TRACKED, configured_field, sha
from noise_injection_v2 import raw_for_unique_ids, score

REGISTRATION = Path("configs/cold_v4_deep_fields.json")
RESULT = Path("data/cold_v4_deep/policy_development.json")
REPEATS = 4
MAX_TEMPLATES = 1000


def raw_for_ids(ids, field):
    """Read source or environment raw rows in requested Legacy-ID order."""
    config = json.loads(Path(field).read_text())
    stage = json.loads(Path(config["base_config"]).read_text())
    unique, inverse = np.unique(np.asarray(ids, np.int64), return_inverse=True)
    if len(unique) != len(ids):
        raise ValueError("Input source or environment Legacy IDs repeat")
    return raw_for_unique_ids(unique, stage["sweep_root"], config["sweep_files"])[inverse]


def source_templates(field, rng):
    references = load_reference_catalogs("data/reference")
    table, cold, mg, metadata = configured_field(field, references)
    eligible = np.flatnonzero(cold & table["full_high_quality"])
    if len(eligible) < 100:
        raise ValueError(f"Only {len(eligible)} full-band high-S/N Gaia templates")
    if len(eligible) > MAX_TEMPLATES:
        eligible = np.sort(rng.choice(eligible, MAX_TEMPLATES, replace=False))
    ids = np.asarray(table["ls_id"][eligible], np.int64)
    source_id = np.asarray(table["source_id"][eligible], np.int64)
    if len(np.unique(ids)) != len(ids) or len(np.unique(source_id)) != len(ids):
        raise ValueError("Template Legacy or Gaia identity repeats")
    raw = raw_for_ids(ids, field)
    template = np.empty(len(raw), dtype=raw.dtype.descr + [("r0", "f8")])
    for name in raw.dtype.names:
        template[name] = raw[name]
    template["r0"] = table["r0"][eligible]
    truth_mg = np.asarray(mg[eligible], float)
    truth_mu = np.asarray(table["g0"][eligible], float) - truth_mg
    if not np.isfinite(truth_mg).all() or not np.isfinite(truth_mu).all():
        raise ValueError("Nonfinite template distance truth")
    return template, truth_mg, truth_mu, source_id, metadata


def load_environment(field):
    config = json.loads(Path(field).read_text())
    path = Path(config["output_dir"]) / "field_diagnostics.fits"
    return fitsio.read(path, columns=["ls_id", "r", "grz_base"])


def frozen_models():
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Published cold-v4 preparation changed")
    prepared = json.loads(PREPARED.read_text())
    if prepared["registration_sha256"] != sha("configs/cold_v4_fields.json"):
        raise ValueError("Cold-v4 registration changed")
    path = OUT / "route.joblib"
    if sha(path) != prepared["route"]["route_sha256"]:
        raise ValueError("Cold-v4 route changed")
    forests = {}
    for tier in FEATURES:
        path = OUT / f"{tier}.joblib"
        if sha(path) != prepared["tier_training"][tier]["model_sha256"]:
            raise ValueError(f"Cold-v4 forest changed: {tier}")
        forests[tier] = joblib.load(path)
    return joblib.load(OUT / "route.joblib"), forests


def direct_predictions(phot, routed, forests):
    """Predict each supported tier once for all candidate policies."""
    predictions = {}
    for tier in FEATURES:
        take = routed["support_by_tier"][tier]
        values = np.full((len(take), 5), np.nan)
        if take.any():
            values[take] = forests[tier].predict(feature_matrix(phot, tier)[take])
        predictions[tier] = values
    return predictions


def evaluate_draw(template, truth_mg, truth_mu, target, route, forests, policies, rng, seed):
    count = len(template)
    raw, shift, ratio = simulate_shared_flux(template, target, rng)
    stage = json.loads(Path("configs/stage1.json").read_text())
    phot = derive_photometry(raw, **stage["photometry"])
    routed = route_rows(phot, route)
    predicted = direct_predictions(phot, routed, forests)
    result = {"draws": count, "simulated_grz_base": int(phot["grz_base"].sum()),
              "simulated_grz_clean": int(phot["grz_clean"].sum()),
              "template_noise_gt_target_fraction": {
                  band: float(np.nanmean(ratio[:, index] > 1))
                  for index, band in enumerate(("g", "r", "z", "w1", "w2"))},
              "policies": {}}
    for order, policy in enumerate(policies):
        level = choose_policy(phot, routed, policy)
        take = level > 0
        row = {"selected": int(take.sum()),
               "acceptance": float(take.mean()),
               "selected_clean": int(np.sum(take & phot["grz_clean"])),
               "chosen_tier_counts": {tier: int(np.sum(level == number))
                                      for tier, number in LEVEL.items()},
               "tiers": {}}
        if take.any():
            pred = np.full((count, 5), np.nan)
            for tier, number in LEVEL.items():
                use = level == number
                pred[use] = predicted[tier][use]
            if not np.isfinite(pred[take]).all():
                raise ValueError("Nonfinite selected policy prediction")
            index = np.arange(count) // REPEATS
            row["direct"] = score(truth_mg[take], truth_mu[take] + shift[take],
                                  np.asarray(phot["g0"][take], float), pred[take],
                                  index[take], seed=seed + order)
            for tier, number in LEVEL.items():
                use = level == number
                if use.sum() >= 10:
                    row["tiers"][tier] = score(
                        truth_mg[use], truth_mu[use] + shift[use],
                        np.asarray(phot["g0"][use], float), pred[use],
                        index[use], seed=seed + order + number * 100)
        result["policies"][policy] = row
    return result


def main():
    require_compute()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    config = json.loads(REGISTRATION.read_text())
    route, forests = frozen_models()
    rng = np.random.default_rng(config["seed"])
    field = config["policy_development"]["field_config"]
    template, mg, mu, source_id, metadata = source_templates(field, rng)
    mother = load_environment(field)
    result = {"registration_sha256": sha(REGISTRATION),
              "prepared_sha256": sha(PREPARED),
              "compute_host": socket.gethostname(),
              "template_field": metadata,
              "independent_template_sources": len(template),
              "template_source_id_sha256": identity_hash(source_id),
              "repeats_per_source_and_depth": REPEATS,
              "same_five_band_draw_for_all_policies": True,
              "environment_mother": "all grz_base rows in role field; no WISE preselection",
              "depths": []}
    for depth_index, (lo, hi) in enumerate(config["depth_bins_r"]):
        pool = mother[mother["grz_base"] & (mother["r"] >= lo) & (mother["r"] < hi)]
        if len(pool) < 100:
            raise ValueError(f"Only {len(pool)} target environments in r={lo}-{hi}")
        source_index = np.repeat(np.arange(len(template)), REPEATS)
        target_index = rng.integers(len(pool), size=len(source_index))
        target_ids = np.asarray(pool["ls_id"][target_index], np.int64)
        unique, inverse = np.unique(target_ids, return_inverse=True)
        target = raw_for_ids(unique, field)[inverse]
        row = evaluate_draw(template[source_index], mg[source_index], mu[source_index],
                            target, route, forests, config["policies"], rng,
                            config["seed"] + depth_index * 1000)
        row["r_bin"] = [lo, hi]
        row["target_mother_rows"] = len(pool)
        result["depths"].append(row)
        print(json.dumps({"r_bin": [lo, hi], "policies": {
            key: value["chosen_tier_counts"]
            for key, value in row["policies"].items()}}), flush=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, result)


if __name__ == "__main__":
    main()

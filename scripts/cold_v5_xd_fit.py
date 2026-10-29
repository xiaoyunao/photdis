#!/usr/bin/env python3
"""Fit one preregistered XD mixture size to frozen-v4 tier identities."""
from __future__ import annotations

import argparse
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
from photdis.route import FEATURES, LAYER, LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from photdis.xd_distance import XDMixture, measurement_covariance
from cold_v4 import (OUT as V4_OUT, PREPARED, PREPARED_TRACKED,
                     PRIOR_CONFIGS, configured_field, original_pilot, sha)

CONFIG = Path("configs/cold_v5_xd_experiment.json")
ERRORS = ("sigma_g", "sigma_r", "sigma_z", "sigma_w1", "sigma_w2")
EXTRA = ("source_id", "snr_corr", *ERRORS, "grz_clean", "snr_w1", "snr_w2", "r",
         "valid_w1", "valid_w2", "w1_clean", "w2_clean")
BANDS = {"A_grz": ("g", "r", "z"),
         "B_grzW1": ("g", "r", "z", "w1"),
         "C_grzW1W2": ("g", "r", "z", "w1", "w2")}


def add_sidecar(table, metadata):
    if "path" not in metadata:
        return table
    with fitsio.FITS(metadata["path"]) as file:
        rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        raw = file[1].read(columns=list(EXTRA), rows=rows)
    if not np.array_equal(raw["source_id"], table["source_id"]):
        raise ValueError("Gaia anchor sidecar identity/order changed")
    table.update({name: raw[name] for name in raw.dtype.names if name != "source_id"})
    return table


def load_training(references):
    pilot_fit, _ = original_pilot(references)
    sources = [pilot_fit]
    for path in PRIOR_CONFIGS:
        table, selected, mg, meta = configured_field(path, references)
        sources.append((add_sidecar(table, meta), selected, mg, meta))
    return sources


def tier_arrays(sources, tier, *, parallax_errors=True):
    values, covariance, identity = [], [], []
    for table, selected, mg, _ in sources:
        colors = feature_matrix(table, tier)
        take = (selected & np.asarray(table[LAYER[tier]], bool) &
                np.isfinite(colors).all(axis=1))
        if not take.any():
            continue
        sigma = np.column_stack([table[f"sigma_{band}"][take]
                                 for band in BANDS[tier]])
        fraction = 1 / np.asarray(table["snr_corr"][take], float) if parallax_errors else None
        covariance.append(measurement_covariance(sigma, fraction))
        values.append(np.column_stack([np.asarray(mg[take], float), colors[take]]))
        identity.append(np.asarray(table["source_id"][take], np.int64))
    return np.concatenate(values), np.concatenate(covariance), np.concatenate(identity)


def load_development(config_path, references, route, policy):
    table, cold, mg, meta = configured_field(config_path, references)
    add_sidecar(table, meta)
    chosen = choose_policy(table, route_rows(table, route), policy)
    return table, cold, mg, chosen, meta


def development_score(model, tier, fields):
    result = {}
    for table, cold, mg, chosen, meta in fields:
        take = cold & (chosen == LEVEL[tier])
        if not take.any():
            result[meta["name"]] = {"n": 0, "mean_negative_conditional_log_density": None}
            continue
        sigma = np.column_stack([table[f"sigma_{band}"][take]
                                 for band in BANDS[tier]])
        covariance = measurement_covariance(sigma,
            1 / np.asarray(table["snr_corr"][take], float))
        values = np.column_stack([np.asarray(mg[take], float),
                                   feature_matrix(table, tier)[take]])
        log_density = model.conditional_mg_logpdf(values, covariance)
        result[meta["name"]] = {"n": int(take.sum()),
            "mean_negative_conditional_log_density": float(-np.mean(log_density))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, required=True)
    args = parser.parse_args()
    require_compute()
    cfg = json.loads(CONFIG.read_text())
    if args.components not in cfg["components"]:
        parser.error("Components not in preregistered comparison")
    if sha(PREPARED) != cfg["v4_prepared_sha256"] or sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen v4 preparation changed")
    prepared = json.loads(PREPARED.read_text())
    route_path = V4_OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen v4 route changed")
    route = joblib.load(route_path)
    output_dir = Path("data/cold_v5_xd") / f"k{args.components:02d}"
    manifest_path = Path("manifests") / f"cold_v5_xd_k{args.components:02d}.json"
    if output_dir.exists() or manifest_path.exists():
        raise FileExistsError("XD result already exists")
    output_dir.mkdir(parents=True)
    references = load_reference_catalogs("data/reference")
    sources = load_training(references)
    development = [load_development(path, references, route, cfg["v4_policy"])
                   for path in cfg["configuration_development_fields"]]
    report = {"components": args.components, "configuration_sha256": sha(CONFIG),
              "prepared_sha256": sha(PREPARED), "tiers": {},
              "development_field_names": [row[4]["name"] for row in development],
              "compute_host": socket.gethostname()}
    for tier, level in LEVEL.items():
        observed, errors, ids = tier_arrays(sources, tier)
        expected = prepared["tier_training"][tier]
        if len(ids) != expected["n_fit"] or identity_hash(ids) != expected["identity_sha256"]:
            raise ValueError(f"Frozen {tier} fit identities changed")
        model = XDMixture(args.components,
            covariance_floor=cfg["covariance_floor_mag2"],
            max_iter=cfg["xd_max_iterations"],
            tolerance=cfg["xd_relative_tolerance"],
            seed=20261029 + args.components + level)
        model.fit(observed, errors)
        path = output_dir / f"{tier}.joblib"
        joblib.dump(model, path)
        report["tiers"][tier] = {
            "fit_rows": len(ids), "identity_sha256": identity_hash(ids),
            "model_sha256": sha(path),
            "iterations": len(model.log_likelihood_history_),
            "first_log_likelihood": model.log_likelihood_history_[0],
            "last_log_likelihood": model.log_likelihood_history_[-1],
            "minimum_covariance_eigenvalue": float(min(np.linalg.eigvalsh(
                model.covariances_).min(axis=1))),
            "development": development_score(model, tier, development)}
        print(json.dumps({"components": args.components, "tier": tier,
              "iterations": report["tiers"][tier]["iterations"],
              "last_log_likelihood": report["tiers"][tier]["last_log_likelihood"]}),
              flush=True)
    write_json(manifest_path, report)
    print(json.dumps({"manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate final photometric distances with a shared five-band noise draw."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photdis.followup import apply_cqr, identity_hash
from photdis.noise import (BANDS, interval_diagnostics, simulate_shared_flux,
                           transformed_distance_quantiles)
from photdis.photometry import derive_photometry
from photdis.stage1 import PHOT_COLUMNS, legacy_ids, require_compute, write_json

OUT = Path("data/noise_injection_v2")
FIG = Path("figures/noise_injection_v2")
FEATURES = {"A_grz": ("gr0", "rz0"),
            "B_grzW1": ("gr0", "rz0", "zw10"),
            "C_grzW1W2": ("gr0", "rz0", "zw10", "w1w20")}
TIERS = {"A_grz": "grz_base", "B_grzW1": "grz_w1", "C_grzW1W2": "grz_w1_w2"}
STRICT = {"A_grz": "grz_high_quality", "B_grzW1": "grz_w1_high_quality",
          "C_grzW1W2": "full_high_quality"}
DEPTHS = ((18, 19), (19, 20), (20, 21), (21, 22))
REPEATS = 8
SEED = 20261028


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_for_unique_ids(ids, sweep_root, sweep_names):
    """Read only selected raw rows after a three-column identity scan."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) == 0 or np.any(np.diff(ids) <= 0):
        raise ValueError("Input Legacy IDs must be sorted and unique")
    result = None
    found = np.zeros(len(ids), bool)
    columns = ["MASKBITS", *PHOT_COLUMNS]
    for name in sweep_names:
        path = Path(sweep_root) / name
        ident = fitsio.read(path, ext=1, columns=["RELEASE", "BRICKID", "OBJID"])
        packed = legacy_ids(ident)
        positions = np.searchsorted(ids, packed)
        hit = positions < len(ids)
        rows = np.flatnonzero(hit)
        hit[rows] = ids[positions[rows]] == packed[rows]
        rows = np.flatnonzero(hit)
        if not len(rows):
            continue
        values = fitsio.read(path, ext=1, rows=rows, columns=columns)
        if result is None:
            result = np.empty(len(ids), dtype=[(key.lower(), values[key].dtype)
                                               for key in columns])
        dest = positions[rows]
        if found[dest].any():
            raise ValueError("Legacy object appears in more than one sweep")
        for key in columns:
            result[key.lower()][dest] = values[key]
        found[dest] = True
    if not found.all():
        raise ValueError(f"Could not recover {int((~found).sum())} target raw rows")
    return result


def source_cluster_ci(truth, quantiles, source_index, seed=SEED, n_boot=300):
    """Cluster repeated draws by original anchor, preserving within-star noise."""
    source_index = np.asarray(source_index)
    groups = np.unique(source_index)
    if len(groups) < 10:
        return None
    lookup = [np.flatnonzero(source_index == group) for group in groups]
    rng = np.random.default_rng(seed)
    cover68, cover95, nmad = [], [], []
    for _ in range(n_boot):
        take = np.concatenate([lookup[i] for i in rng.integers(len(groups), size=len(groups))])
        result = interval_diagnostics(truth[take], quantiles[take])
        cover68.append(result["coverage_68"])
        cover95.append(result["coverage_95"])
        nmad.append(result["nmad"])
    return {"cluster": "original Gaia anchor source", "n_groups": len(groups),
            "coverage_68_95pct": np.quantile(cover68, [.025, .975]).tolist(),
            "coverage_95_95pct": np.quantile(cover95, [.025, .975]).tolist(),
            "nmad_95pct": np.quantile(nmad, [.025, .975]).tolist()}


def score(truth_mg, truth_mu, g0, predicted, source_index, seed):
    mu, distance = transformed_distance_quantiles(g0, predicted)
    truth_distance = 10 ** ((truth_mu + 5) / 5)
    mg_result = interval_diagnostics(truth_mg, predicted)
    mu_result = interval_diagnostics(truth_mu, mu)
    distance_result = interval_diagnostics(truth_distance, distance)
    distance_result["median_relative_error"] = float(np.median(distance[:, 2] / truth_distance - 1))
    distance_result["nmad_relative_error"] = float(1.4826 * np.median(
        np.abs(distance[:, 2] / truth_distance - 1 - distance_result["median_relative_error"])))
    return {"absolute_magnitude_mag": mg_result,
            "distance_modulus_mag": mu_result,
            "distance_pc": distance_result,
            "distance_modulus_source_cluster_ci": source_cluster_ci(
                truth_mu, mu, source_index, seed=seed)}


def main():
    require_compute()
    if (OUT / "summary.json").exists():
        raise FileExistsError(OUT / "summary.json")
    frozen = json.loads(Path("configs/followup_fields.json").read_text())
    for path, digest in (("manifests/cold_baseline.json", frozen["baseline_manifest_sha256"]),
                         ("data/cold_baseline_v3/test_predictions.fits",
                          frozen["baseline_test_predictions_sha256"])):
        if sha(path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    prediction = fitsio.read("data/cold_baseline_v3/test_predictions.fits")
    reference = fitsio.read("data/pilot_audit_v1/reference.fits")
    common_ids = prediction["source_id"][np.isfinite(prediction["C_grzW1W2_full_q500"])]
    anchor = reference[np.isin(reference["source_id"], common_ids)]
    if len(anchor) != len(common_ids) or not np.array_equal(np.sort(anchor["source_id"]), np.sort(common_ids)):
        raise ValueError("Common frozen A/B/C test identities changed")
    for name in FEATURES:
        if sha(f"data/cold_baseline_v3/{name}.joblib") != frozen["baseline_model_sha256"][name]:
            raise ValueError(f"Frozen model changed: {name}")
    models = {name: joblib.load(f"data/cold_baseline_v3/{name}.joblib") for name in FEATURES}
    offsets = json.loads(Path("manifests/external_validation.json").read_text())["offsets_mag"]
    pilot = json.loads(Path("configs/pilot_10deg.json").read_text())
    stage = json.loads(Path("configs/stage1.json").read_text())
    diagnostic = fitsio.read("data/pilot_10deg_v1/field_diagnostics.fits",
                             columns=["ls_id", "r", "grz_base"])
    truth_mg = np.asarray(anchor["g0"] - 10 + 5 * np.log10(anchor["p_corr"]), float)
    truth_mu = np.asarray(10 - 5 * np.log10(anchor["p_corr"]), float)
    rng = np.random.default_rng(SEED)
    summary = {"frozen_common_test_id_sha256": identity_hash(anchor["source_id"]),
               "common_test_sources": len(anchor), "repeats_per_source_and_depth": REPEATS,
               "target_mother": "Original 10-degree field PSF rows with grz_base; no W1/W2 detection or quality preselection",
               "target_mother_rows": int(np.sum(diagnostic["grz_base"])),
               "target_noise": "Raw DR11 flux inverse variances and target measurement flags, including missing W1/W2; same sampled row and Gaussian draw for every model",
               "template_noise": "Existing anchor measurement noise reduced in quadrature from target Gaussian draw when possible; ratios and infeasible cases reported",
               "truth": "Gaia corrected parallax plus original anchor grz label; true modulus shifted to observed target r0",
               "compute_host": socket.gethostname(), "depths": []}
    for depth_index, (lo, hi) in enumerate(DEPTHS):
        mother = diagnostic[diagnostic["grz_base"] & (diagnostic["r"] >= lo) &
                            (diagnostic["r"] < hi)]
        if len(mother) < 20:
            summary["depths"].append({"r_bin": [lo, hi], "mother_rows": len(mother),
                                      "status": "insufficient_grz_mother"})
            continue
        source_index = np.repeat(np.arange(len(anchor)), REPEATS)
        target_index = rng.integers(len(mother), size=len(source_index))
        sampled_ids = mother["ls_id"][target_index]
        unique_ids, inverse = np.unique(sampled_ids, return_inverse=True)
        target = raw_for_unique_ids(unique_ids, stage["sweep_root"], pilot["sweep_files"])[inverse]
        repeated = anchor[source_index]
        raw, shift, noise_ratio = simulate_shared_flux(repeated, target, rng)
        phot = derive_photometry(raw, **stage["photometry"])
        true_mg = truth_mg[source_index]
        true_mu = truth_mu[source_index] + shift
        row = {"r_bin": [lo, hi], "mother_rows": len(mother), "draws": len(source_index),
               "template_to_target_sigma_ratio_median": {
                   band.lower(): float(np.nanmedian(noise_ratio[:, index]))
                   for index, band in enumerate(BANDS)},
               "template_noise_gt_target_fraction": {
                   band.lower(): float(np.nanmean(noise_ratio[:, index] > 1))
                   for index, band in enumerate(BANDS)},
               "tiers": {}}
        full_predictions = {}
        for tier_index, (tier, features) in enumerate(FEATURES.items()):
            accepted = phot[TIERS[tier]] & np.logical_and.reduce(
                [np.isfinite(phot[name]) for name in features])
            strict = accepted & phot[STRICT[tier]]
            tier_row = {"accepted": int(accepted.sum()),
                        "acceptance_fraction_of_grz_mother": float(accepted.mean()),
                        "strict_quality": int(strict.sum()),
                        "strict_quality_fraction_of_grz_mother": float(strict.mean())}
            if accepted.any():
                x = np.column_stack([phot[name][accepted] for name in features])
                predicted = models[tier].predict(x)
                calibrated = apply_cqr(predicted, offsets[tier]["q68"], offsets[tier]["q95"])
                full_predictions[tier] = np.full((len(source_index), 5), np.nan)
                full_predictions[tier][accepted] = calibrated
                g0 = np.asarray(phot["g0"][accepted], float)
                sample = (true_mg[accepted], true_mu[accepted], g0,
                          source_index[accepted])
                tier_row["raw"] = score(*sample[:3], predicted, sample[3],
                                         seed=SEED + depth_index * 10 + tier_index)
                tier_row["external_cqr"] = score(*sample[:3], calibrated, sample[3],
                                                  seed=SEED + depth_index * 10 + tier_index)
            row["tiers"][tier] = tier_row
        common = (np.logical_and.reduce([np.isfinite(full_predictions[name][:, 2])
                                         for name in FEATURES])
                  if len(full_predictions) == len(FEATURES)
                  else np.zeros(len(source_index), bool))
        row["same_accepted_all_three"] = int(common.sum())
        if len(full_predictions) == len(FEATURES) and common.any():
            row["same_accepted_tier_distance_modulus"] = {}
            for tier_index, tier in enumerate(FEATURES):
                evaluated = score(true_mg[common], true_mu[common],
                                  np.asarray(phot["g0"][common], float),
                                  full_predictions[tier][common], source_index[common],
                                  seed=SEED + depth_index * 10 + tier_index + 100)
                row["same_accepted_tier_distance_modulus"][tier] = evaluated["distance_modulus_mag"]
        summary["depths"].append(row)
        print(json.dumps({"r_bin": [lo, hi], "mother": len(mother),
                          "acceptance": {name: item["acceptance_fraction_of_grz_mother"]
                                         for name, item in row["tiers"].items()}}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    depths = [item for item in summary["depths"] if "tiers" in item]
    x = [sum(item["r_bin"]) / 2 for item in depths]
    for tier, color in zip(FEATURES, ("#356e9b", "#58894e", "#bf6f38")):
        rows = [item["tiers"][tier] for item in depths]
        axes[0].plot(x, [r["acceptance_fraction_of_grz_mother"] for r in rows], "o-",
                     color=color, label=tier)
        axes[1].plot(x, [r["strict_quality_fraction_of_grz_mother"] for r in rows],
                     "o-", color=color)
        axes[2].plot(x, [r.get("external_cqr", {}).get("distance_modulus_mag", {})
                         .get("coverage_95", np.nan) for r in rows], "o-", color=color)
    axes[2].axhline(.95, color=".5", ls=":")
    for ax in axes:
        ax.set_xlabel("Target observed r [mag]")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Input acceptance / grz mother")
    axes[1].set_ylabel("Strict quality / grz mother")
    axes[2].set_ylabel("Final distance modulus 95% coverage")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_shared_noise_final_distance.png", dpi=170)
    plt.close(fig)
    write_json(OUT / "summary.json", summary)
    write_json("manifests/noise_injection_v2.json", summary)


if __name__ == "__main__":
    main()

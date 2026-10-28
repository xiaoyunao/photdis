#!/usr/bin/env python3
"""Apply the calibrated cold-v4 route to one shared five-band noise draw."""
from __future__ import annotations

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

from photdis.conditional import group_labels
from photdis.followup import identity_hash
from photdis.noise import BANDS, simulate_shared_flux
from photdis.photometry import derive_photometry
from photdis.route import FEATURES, LEVEL, apply_route_offsets, feature_matrix, route_rows
from photdis.stage1 import require_compute, write_json
from cold_v4 import OUT, PREPARED, PREPARED_TRACKED, registration, sha
from noise_injection_v2 import (DEPTHS, REPEATS, SEED, STRICT,
                                raw_for_unique_ids, score)

OUTPUT = OUT / "noise_summary.json"
FIGURE = Path("figures/cold_v4/01_route_shared_noise.png")


def main():
    require_compute()
    if OUTPUT.exists() or FIGURE.exists():
        raise FileExistsError("Cold-v4 noise products already exist")
    if sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Published cold-v4 preparation changed")
    _, frozen = registration()
    prepared = json.loads(PREPARED.read_text())
    if prepared["registration_sha256"] != sha("configs/cold_v4_fields.json"):
        raise ValueError("Cold-v4 registration changed")
    route_path = OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Cold-v4 route changed")
    route = joblib.load(route_path)
    forests = {}
    for tier in FEATURES:
        path = OUT / f"{tier}.joblib"
        if sha(path) != prepared["tier_training"][tier]["model_sha256"]:
            raise ValueError(f"Cold-v4 forest changed: {tier}")
        forests[tier] = joblib.load(path)
    for path, digest in (("data/cold_baseline_v3/test_predictions.fits",
                          frozen["baseline_test_predictions_sha256"]),):
        if sha(path) != digest:
            raise ValueError(f"Frozen test changed: {path}")
    original_prediction = fitsio.read("data/cold_baseline_v3/test_predictions.fits")
    reference = fitsio.read("data/pilot_audit_v1/reference.fits")
    common_ids = original_prediction["source_id"][np.isfinite(
        original_prediction["C_grzW1W2_full_q500"])]
    anchor = reference[np.isin(reference["source_id"], common_ids)]
    if len(anchor) != len(common_ids) or identity_hash(anchor["source_id"]) != identity_hash(common_ids):
        raise ValueError("Frozen common noise-template identities changed")
    truth_mg = np.asarray(anchor["g0"] - 10 + 5 * np.log10(anchor["p_corr"]), float)
    truth_mu = np.asarray(10 - 5 * np.log10(anchor["p_corr"]), float)
    pilot = json.loads(Path("configs/pilot_10deg.json").read_text())
    stage = json.loads(Path("configs/stage1.json").read_text())
    diagnostic = fitsio.read("data/pilot_10deg_v1/field_diagnostics.fits",
                             columns=["ls_id", "r", "grz_base"])
    edges = np.asarray(prepared["route_groups"]["color_edges"], float)
    r_edges = prepared["route_groups"]["r_edges_by_color"]
    offsets = prepared["calibration_offsets"]
    rng = np.random.default_rng(SEED)
    summary = {"prepared_sha256": sha(PREPARED),
               "frozen_test_identity_sha256": identity_hash(anchor["source_id"]),
               "n_independent_templates": len(anchor),
               "repeats_per_template_and_depth": REPEATS,
               "target_mother": "all grz_base rows in the original pilot, without WISE success preselection",
               "target_mother_rows": int(diagnostic["grz_base"].sum()),
               "same_five_band_noise_draw_for_all_tiers": True,
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
        selected_ids = mother["ls_id"][target_index]
        ids, inverse = np.unique(selected_ids, return_inverse=True)
        target = raw_for_unique_ids(ids, stage["sweep_root"], pilot["sweep_files"])[inverse]
        raw, shift, noise_ratio = simulate_shared_flux(anchor[source_index], target, rng)
        phot = derive_photometry(raw, **stage["photometry"])
        routed = route_rows(phot, route)
        selected = routed["selected"]
        strict = np.zeros(len(selected), bool)
        direct = np.full((len(selected), 5), np.nan)
        tier_counts = {}
        for tier, level in LEVEL.items():
            take = selected & (routed["tier"] == level)
            tier_counts[tier] = int(take.sum())
            strict[take] = phot[STRICT[tier]][take]
            if take.any():
                direct[take] = forests[tier].predict(feature_matrix(phot, tier)[take])
        if not np.isfinite(direct[selected]).all():
            raise ValueError("Nonfinite noise prediction")
        row = {"r_bin": [lo, hi], "mother_rows": len(mother),
               "draws": len(source_index),
               "simulated_grz_base": int(phot["grz_base"].sum()),
               "color_box": int(routed["color_box"].sum()),
               "selected": int(selected.sum()),
               "selected_fraction_of_grz_mother": float(selected.mean()),
               "strict_selected": int(strict.sum()),
               "strict_fraction_of_grz_mother": float(strict.mean()),
               "chosen_tier_counts": tier_counts,
               "template_noise_gt_target_fraction": {
                   band.lower(): float(np.nanmean(noise_ratio[:, index] > 1))
                   for index, band in enumerate(BANDS)}}
        if selected.any():
            groups = group_labels(phot["rz0"][selected], phot["r"][selected],
                                  edges, r_edges)
            adjusted = apply_route_offsets(direct[selected], groups, offsets)
            truth_mg_selected = truth_mg[source_index][selected]
            truth_mu_selected = truth_mu[source_index][selected] + shift[selected]
            g0 = np.asarray(phot["g0"][selected], float)
            indices = source_index[selected]
            row["direct"] = score(truth_mg_selected, truth_mu_selected, g0,
                                  direct[selected], indices, seed=SEED + depth_index)
            row["calibrated"] = score(truth_mg_selected, truth_mu_selected, g0,
                                      adjusted, indices, seed=SEED + depth_index + 100)
            true_distance = 10 ** ((truth_mu_selected + 5) / 5)
            distance_median = 10 ** ((g0 - adjusted[:, 2] + 5) / 5)
            row["large_relative_distance_abs_gt_0p5"] = float(np.mean(
                np.abs(distance_median / true_distance - 1) > .5))
        summary["depths"].append(row)
        print(json.dumps({"r_bin": [lo, hi], "selected": row["selected"],
                          "strict": row["strict_selected"],
                          "tiers": tier_counts}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, summary)

    rows = [item for item in summary["depths"] if "calibrated" in item]
    x = [sum(row["r_bin"]) / 2 for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    axes[0].plot(x, [row["selected_fraction_of_grz_mother"] for row in rows], "o-",
                 label="complete route")
    axes[0].plot(x, [row["strict_fraction_of_grz_mother"] for row in rows], "s--",
                 label="route + strict quality")
    axes[1].plot(x, [row["calibrated"]["distance_modulus_mag"]["coverage_95"]
                     for row in rows], "o-")
    axes[1].axhline(.95, color=".5", ls=":")
    axes[0].set_ylabel("Fraction of grz mother")
    axes[1].set_ylabel("Final distance-modulus 95% coverage")
    axes[1].set_ylim(0, 1)
    axes[0].legend()
    for ax in axes:
        ax.set_xlabel("Target observed r [mag]")
        ax.grid(alpha=.2)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

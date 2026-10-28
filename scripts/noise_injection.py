#!/usr/bin/env python3
"""Transfer frozen pilot test stars to observed target depths in flux space."""
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

from photdis.photometry import derive_photometry, MAG_ERROR_FACTOR
from photdis.quantile import metrics
from photdis.stage1 import PHOT_COLUMNS, require_compute, write_json
from photdis.followup import apply_cqr, identity_hash

OUT = Path("data/noise_injection_v1")
FIG = Path("figures/noise_injection")
FEATURES = {"A_grz": ("gr0", "rz0"),
            "B_grzW1": ("gr0", "rz0", "zw10"),
            "C_grzW1W2": ("gr0", "rz0", "zw10", "w1w20")}
TIERS = {"A_grz": "grz_base", "B_grzW1": "grz_w1", "C_grzW1W2": "grz_w1_w2"}
DEPTHS = ((18, 19), (19, 20), (20, 21), (21, 22))
REPEATS = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulate(anchor, target, rng):
    """Shift all source fluxes by one modulus, then draw target-sized band noise."""
    raw = np.empty(len(anchor), dtype=[(name, anchor[name.lower()].dtype)
                                        for name in PHOT_COLUMNS])
    for name in PHOT_COLUMNS:
        raw[name] = anchor[name.lower()]
    shift = 10 ** (-.4 * (target["r"].astype(float) - anchor["r"].astype(float)))
    for band in ("G", "R", "Z", "W1", "W2"):
        lower = band.lower()
        old_flux = anchor[f"flux_{lower}"].astype(float) * shift
        target_flux = 10 ** ((22.5 - target[lower].astype(float)) / 2.5)
        noise_sigma = target_flux * target[f"sigma_{lower}"].astype(float) / MAG_ERROR_FACTOR
        usable = np.isfinite(noise_sigma) & (noise_sigma > 0) & np.isfinite(old_flux)
        raw[f"FLUX_{band}"][usable] = (old_flux[usable] +
                                          rng.normal(size=usable.sum()) * noise_sigma[usable])
        raw[f"FLUX_IVAR_{band}"][usable] = 1 / noise_sigma[usable] ** 2
    return raw


def main():
    require_compute()
    if (OUT / "summary.json").exists():
        raise FileExistsError(OUT / "summary.json")
    frozen = json.loads(Path("configs/followup_fields.json").read_text())
    for path, digest in (("manifests/cold_baseline.json", frozen["baseline_manifest_sha256"]),
                         ("data/cold_baseline_v3/test_predictions.fits",
                          frozen["baseline_test_predictions_sha256"])):
        if sha(path) != digest:
            raise ValueError(f"Frozen test input changed: {path}")
    prediction = fitsio.read("data/cold_baseline_v3/test_predictions.fits")
    reference = fitsio.read("data/pilot_audit_v1/reference.fits")
    target_columns = ["r", "grz_base", "grz_w1", "grz_w1_w2"] + [
        item for band in ("g", "r", "z", "w1", "w2")
        for item in (band, f"sigma_{band}")]
    target_columns = list(dict.fromkeys(target_columns))
    target = fitsio.read("data/pilot_10deg_v1/field_diagnostics.fits", columns=target_columns)
    phot_config = json.loads(Path("configs/stage1.json").read_text())["photometry"]
    calibration = json.loads(Path("manifests/external_validation.json").read_text())["offsets_mag"]
    rng = np.random.default_rng(20261028)
    summary = {"frozen_test_id_sha256": identity_hash(prediction["source_id"]),
               "repeats_per_source_and_depth": REPEATS,
               "flux_noise": "Independent Gaussian per band, sigma from an observed target row; common distance-modulus shift preserves intrinsic colors and induces their correct covariance",
               "target_field": "original pilot; grz measured target rows",
               "depth_bins_r_observed": [list(pair) for pair in DEPTHS],
               "tiers": {}, "compute_host": socket.gethostname()}
    for tier, names in FEATURES.items():
        model_path = f"data/cold_baseline_v3/{tier}.joblib"
        if sha(model_path) != frozen["baseline_model_sha256"][tier]:
            raise ValueError(f"Frozen model changed: {tier}")
        model = joblib.load(model_path)
        allowed_ids = prediction["source_id"][np.isfinite(prediction[f"{tier}_full_q500"])]
        anchor = reference[np.isin(reference["source_id"], allowed_ids)]
        if len(anchor) != len(allowed_ids) or not np.array_equal(np.sort(anchor["source_id"]), np.sort(allowed_ids)):
            raise ValueError(f"Frozen {tier} test identities changed")
        truth = anchor["g0"] - 10 + 5 * np.log10(anchor["p_corr"])
        rows = []
        for lo, hi in DEPTHS:
            population = target[(target[TIERS[tier]]) & (target["r"] >= lo) & (target["r"] < hi)]
            if len(population) < 20:
                rows.append({"r_observed": [lo, hi], "target_rows": len(population), "status": "insufficient_target_rows"})
                continue
            repeated = np.repeat(anchor, REPEATS)
            sampled = population[rng.integers(len(population), size=len(repeated))]
            raw = simulate(repeated, sampled, rng)
            phot = derive_photometry(raw, **phot_config)
            accepted = phot[TIERS[tier]] & np.logical_and.reduce(
                [np.isfinite(phot[name]) for name in names])
            strict_flag = {"A_grz": "grz_high_quality", "B_grzW1": "grz_w1_high_quality",
                           "C_grzW1W2": "full_high_quality"}[tier]
            if not accepted.any():
                rows.append({"r_observed": [lo, hi], "target_rows": len(population),
                             "draws": len(repeated), "accepted": 0,
                             "acceptance_fraction": 0.0, "status": "no_accepted_draws"})
                continue
            x = np.column_stack([phot[name][accepted] for name in names])
            estimates = model.predict(x)
            offset = calibration[tier]
            adjusted = apply_cqr(estimates, offset["q68"], offset["q95"])
            truth_rep = np.repeat(truth, REPEATS)[accepted]
            row = {"r_observed": [lo, hi], "target_rows": len(population),
                   "draws": len(repeated), "accepted": int(accepted.sum()),
                   "acceptance_fraction": float(accepted.mean()),
                   "strict_quality_fraction": float(phot[strict_flag].mean()),
                   "median_target_sigma_mag": {band: float(np.nanmedian(sampled[f"sigma_{band}"]))
                                               for band in ("g", "r", "z", "w1", "w2")},
                   "raw": metrics(truth_rep, estimates),
                   "calibrated": metrics(truth_rep, adjusted)}
            rows.append(row)
            print(json.dumps({"tier": tier, "r": [lo, hi], "accepted": row["accepted"],
                              "raw": row["raw"], "calibrated": row["calibrated"]}), flush=True)
        summary["tiers"][tier] = {"frozen_test_sources": len(anchor), "depths": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for tier, color in zip(FEATURES, ("#356e9b", "#58894e", "#bf6f38")):
        rows = [r for r in summary["tiers"][tier]["depths"] if "raw" in r]
        x = [sum(r["r_observed"]) / 2 for r in rows]
        axes[0].plot(x, [r["acceptance_fraction"] for r in rows], "o-", color=color, label=tier)
        axes[1].plot(x, [r["raw"]["nmad_mag"] for r in rows], "o-", color=color)
        axes[2].plot(x, [r["calibrated"]["coverage_95"] for r in rows], "o-", color=color)
    axes[2].axhline(.95, color=".5", ls=":")
    for ax in axes:
        ax.set_xlabel("Simulated observed r [mag]")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Recomputed input acceptance")
    axes[1].set_ylabel("$M_g$ NMAD [mag]")
    axes[2].set_ylabel("CQR 95% coverage")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_noise_at_target_depths.png", dpi=170)
    plt.close(fig)
    write_json(OUT / "summary.json", summary)
    write_json("manifests/noise_injection.json", summary)


if __name__ == "__main__":
    main()

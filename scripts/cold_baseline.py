#!/usr/bin/env python3
"""Fit and validate a bounded, Gaia-anchored cold main-sequence baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import healpy as hp
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.quantile import QUANTILES, QuantileForest, metrics, spatial_partition
from photdis.stage1 import make_partition_map, require_compute, structured, write_json

OUT = Path("data/cold_baseline_v1")
FIG = Path("figures/cold_baseline")
MANIFEST = Path("manifests/cold_baseline.json")
FEATURES = {"A_grz": ("gr0", "rz0"),
            "B_grzW1": ("gr0", "rz0", "zw10"),
            "C_grzW1W2": ("gr0", "rz0", "zw10", "w1w20")}
TIERS = {"A_grz": "grz_base", "B_grzW1": "grz_w1",
         "C_grzW1W2": "grz_w1_w2"}
LEAVES = (10, 20, 40)


def base_sample(table, external):
    color = table["bp_rp_checked"]
    mg = table["g0"] - 10 + 5 * np.log10(table["p_corr"])
    known_giant = ((table["desi_quality"] & (table["desi_logg"] < 3.5)) |
                   (table["apogee_quality"] & (table["apogee_logg"] < 3.5)))
    mask = (table["anchor"] & ~external & table["bp_rp_precise"] &
            (color >= 1.3) & (color < 3.5) & (mg > 5) & ~known_giant &
            np.isfinite(mg))
    return mask, mg


def feature_matrix(table, names):
    return np.column_stack([table[name].astype(float) for name in names])


def evaluate(table, mg, selection, prediction):
    return metrics(mg[selection], prediction, table["g0"][selection],
                   table["p_corr"][selection])


def binned_diagnostics(table, mg, selection, prediction):
    result = {}
    variables = {"Gaia_BP_RP": (table["bp_rp_checked"], [1.3, 1.8, 2.3, 2.8, 3.5]),
                 "Gaia_G": (table["gaia_phot_g_mean_mag"], [14, 16, 17, 18, 20]),
                 "EBV": (table["ebv"], [0, .01, .02, .03]),
                 "sigma_g": (table["sigma_g"], [0, .01, .02, .03]),
                 "DESI_logg": (table["desi_logg"], [3.5, 4.3, 4.7, 5.5])}
    for name, (value, edges) in variables.items():
        subset = value[selection]
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            take = (subset >= lo) & (subset < hi)
            if take.sum() < 5:
                rows.append({"lo": lo, "hi": hi, "n": int(take.sum())})
            else:
                one = metrics(mg[selection][take], prediction[take])
                rows.append({"lo": lo, "hi": hi, **one})
        result[name] = rows
    return result


def empirical_relation(table, mg, partition, mask):
    train = mask & (partition == 0)
    test = mask & (partition == 2)
    polynomial = np.polynomial.Polynomial.fit(table["rz0"][train], mg[train], 3)
    error = mg[train] - polynomial(table["rz0"][train])
    quantile_error = np.quantile(error, QUANTILES)
    predicted = polynomial(table["rz0"][test])[:, None] + quantile_error
    return metrics(mg[test], predicted, table["g0"][test], table["p_corr"][test]), {
        "coefficient_power_basis": polynomial.convert().coef.tolist(),
        "train_residual_quantiles": quantile_error.tolist()}


def train_tier(table, mg, partition, mask, names, prefix):
    x = feature_matrix(table, names)
    train = mask & (partition == 0)
    validation = mask & (partition == 1)
    test = mask & (partition == 2)
    comparisons = []
    for leaf in LEAVES:
        model = QuantileForest(min_samples_leaf=leaf).fit(x[train], mg[train])
        prediction = model.predict(x[validation])
        score = evaluate(table, mg, validation, prediction)
        objective = (score["nmad_mag"] + abs(score["bias_mag"]) +
                     .5 * abs(score["coverage_68"] - .68) +
                     .25 * abs(score["coverage_95"] - .95))
        comparisons.append({"leaf": leaf, "objective": float(objective), **score})
    chosen = min(comparisons, key=lambda item: item["objective"])["leaf"]
    fit = mask & (partition != 2)
    model = QuantileForest(min_samples_leaf=chosen).fit(x[fit], mg[fit])
    test_prediction = model.predict(x[test])
    if not np.all(np.diff(test_prediction, axis=1) >= -1e-9):
        raise ValueError("Forest quantiles are not ordered")
    joblib.dump(model, OUT / f"{prefix}.joblib")
    return {"features": list(names), "n_train": int(train.sum()),
            "n_validation": int(validation.sum()), "n_test": int(test.sum()),
            "leaf_validation": comparisons, "chosen_leaf": chosen,
            "test": evaluate(table, mg, test, test_prediction),
            "binned_test": binned_diagnostics(table, mg, test, test_prediction)}, test, test_prediction, model


def save_figures(table, mg, selected, partition, results, predictions, common):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    color = table["bp_rp_checked"]
    for part, label, shade in [(0, "train", "#437ca6"), (1, "validation", "#c78c35"),
                               (2, "spatial test", "#c5544b")]:
        mask = selected & (partition == part)
        axes[0].scatter(color[mask], mg[mask], s=6, alpha=.45, label=f"{label}: {mask.sum():,}", color=shade)
        axes[1].scatter(table["ra"][mask], table["dec"][mask], s=6, alpha=.45, color=shade)
    axes[0].set(xlabel="Validated Gaia BP-RP", ylabel="$M_g$ (Legacy + Gaia)", ylim=(15, 4.5))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="RA [deg]", ylabel="Dec [deg]", xlim=(0, 10), ylim=(-10, 0))
    labels = ["grz", "grzW1", "grzW1W2"]
    counts = [results[k]["n_test"] for k in FEATURES]
    axes[2].bar(labels, counts, color=["#437ca6", "#6f9c73", "#a781ae"])
    axes[2].set(ylabel="Test anchors", title="Each tier's full sample")
    fig.suptitle("Cold-main-sequence selection and frozen spatial test")
    fig.tight_layout(); fig.savefig(FIG / "01_sample_and_split.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, (name, (mask, pred)) in zip(axes, predictions.items()):
        residual = pred[:, 2] - mg[mask]
        ax.scatter(table["bp_rp_checked"][mask], residual, s=12, alpha=.7)
        ax.axhline(0, color=".3")
        ax.set(xlabel="Gaia BP-RP", title=f"{name}: N={mask.sum():,}", ylim=(-2, 2))
    axes[0].set_ylabel("Predicted - Gaia $M_g$ [mag]")
    fig.suptitle("Held-out spatial cells: conditional median residuals")
    fig.tight_layout(); fig.savefig(FIG / "02_test_residuals.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for name, (mask, pred) in common.items():
        residual = pred[:, 2] - mg[mask]
        axes[0].hist(residual, bins=np.linspace(-1.5, 1.5, 51), histtype="step", label=name)
        m = results[name]["common_test"]
        axes[1].scatter(m["nmad_mag"], m["coverage_68"], label=name, s=60)
    axes[0].set(xlabel="Predicted - Gaia $M_g$ [mag]", ylabel="Test stars")
    axes[0].legend(fontsize=8)
    axes[1].axhline(.68, color=".5", ls=":")
    axes[1].set(xlabel="Residual NMAD [mag]", ylabel="Actual 68% coverage", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    fig.suptitle("Same-star A/B/C comparison")
    fig.tight_layout(); fig.savefig(FIG / "03_same_star_comparison.png", dpi=170); plt.close(fig)


def main():
    require_compute()
    if (OUT / "summary.json").exists():
        raise FileExistsError(OUT / "summary.json")
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    table = fitsio.read("data/pilot_audit_v1/reference.fits", ext=1)
    audit = json.loads(Path("manifests/pilot_audit.json").read_text())
    if len(table) != audit["reference_count"] or int(table["anchor"].sum()) != 1802:
        raise ValueError("Pilot audit source identity changed")
    holdout = annotate_holdouts(load_reference_catalogs("data/reference"),
                                table["source_id"], table["gaia_ra"], table["gaia_dec"])
    external = holdout["external_any"]
    selected, mg = base_sample(table, external)
    original_cfg = json.loads(Path("configs/stage1.json").read_text())["split"]
    original_cells = hp.ang2pix(8, table["gaia_ra"], table["gaia_dec"], nest=True, lonlat=True)
    original_partition = make_partition_map(original_cfg)[original_cells]
    cells = hp.ang2pix(64, table["gaia_ra"], table["gaia_dec"], nest=True, lonlat=True)
    partition, cell_map = spatial_partition(cells, selected)
    if np.any((original_partition[selected] == 2) & (partition[selected] != 2)):
        raise ValueError("Originally reserved test object changed partition")
    tier_masks = {name: selected & table[flag] & np.all(np.isfinite(feature_matrix(table, FEATURES[name])), axis=1)
                  for name, flag in TIERS.items()}
    common_mask = tier_masks["C_grzW1W2"]
    summary = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "job_id": __import__("os").environ["SLURM_JOB_ID"], "compute_host": socket.gethostname(),
               "selection": {"Gaia_anchor": True, "external_reserved_excluded": int((table["anchor"] & external).sum()),
                             "Gaia_BP_RP": "valid BP/RP flux, NOBS and color error <0.1; 1.3<=BP-RP<3.5",
                             "CMD": "Legacy g0 + corrected Gaia parallax M_g>5",
                             "known_spectroscopic_giant": "exclude logg<3.5",
                             "strict_optical": True},
               "selected_count": int(selected.sum()),
               "original_nside8_partition_counts": np.bincount(original_partition[selected], minlength=4).tolist(),
               "split": {"nside": 64, "ordering": "NESTED", "seed": 20261028,
                         "counts": np.bincount(partition[selected], minlength=4).tolist(), "cell_map": cell_map},
               "label_uncertainty_median_mag": float(np.median(np.hypot(table["sigma_g"][selected],
                                  5 / np.log(10) / table["snr_corr"][selected]))),
               "tiers": {}, "empirical_rz_cubic": None,
               "common_sample": {"counts": np.bincount(partition[common_mask], minlength=4).tolist()},
               "recovered_ps1_diagnostic": {}}
    summary["empirical_rz_cubic"], coefficients = empirical_relation(table, mg, partition, tier_masks["A_grz"])
    summary["empirical_rz_cubic_coefficients"] = coefficients
    predictions = {}
    for name, features in FEATURES.items():
        result, test, pred, model = train_tier(table, mg, partition, tier_masks[name], features, name)
        result["target_count_in_pilot"] = audit["layer_counts"][name.split("_", 1)[1]]["target"]
        result["all_eligible"] = int(tier_masks[name].sum())
        summary["tiers"][name] = result
        predictions[name] = (test, pred)
        if name == "A_grz":
            group = table["audit_group"]
            ps1 = np.logical_and.reduce([table[f"ps1_{b}_quality"] & table[f"valid_{b}"] for b in "grz"])
            delta = np.max(np.column_stack([np.abs(table[f"delta_{b}"]) for b in "grz"]), axis=1)
            recovered = ((group >= 1) & (group <= 3) & ps1 & (delta < .05) &
                         ~external & table["bp_rp_precise"] &
                         (table["bp_rp_checked"] >= 1.3) & (table["bp_rp_checked"] < 3.5) &
                         (mg > 5) & table["grz_base"])
            ext_pred = model.predict(feature_matrix(table, features)[recovered])
            summary["recovered_ps1_diagnostic"] = {"n": int(recovered.sum()),
                     "all_three_PS1_delta_lt_0p05": True,
                     "test_with_strict_A_model": evaluate(table, mg, recovered, ext_pred)}
    common = {}
    for name, features in FEATURES.items():
        fit = common_mask & (partition != 2)
        test = common_mask & (partition == 2)
        model = QuantileForest(min_samples_leaf=20).fit(feature_matrix(table, features)[fit], mg[fit])
        pred = model.predict(feature_matrix(table, features)[test])
        summary["tiers"][name]["common_test"] = evaluate(table, mg, test, pred)
        common[name] = test, pred
    save_figures(table, mg, selected, partition, summary["tiers"], predictions, common)
    fitsio.write(str(OUT / "test_predictions.fits"), structured({
        "source_id": table["source_id"][selected & (partition == 2)],
        "mg_gaia": mg[selected & (partition == 2)],
        "g0": table["g0"][selected & (partition == 2)],
        "bp_rp": table["bp_rp_checked"][selected & (partition == 2)]}),
        extname="TEST", clobber=False)
    summary["elapsed_seconds"] = time.time() - started
    write_json(OUT / "summary.json", summary)
    write_json(MANIFEST, summary)
    print(json.dumps({"selected": summary["selected_count"], "split": summary["split"]["counts"],
                      "tier_tests": {name: item["test"] for name, item in summary["tiers"].items()},
                      "recovered": summary["recovered_ps1_diagnostic"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

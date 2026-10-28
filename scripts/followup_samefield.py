#!/usr/bin/env python3
"""Reconcile bright-star recovery against frozen cold-dwarf test identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import healpy as hp
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from photdis.audit import rvs_metallicity, stats
from photdis.followup import common_nonoptical, identity_hash, ps1_replacement
from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.quantile import QuantileForest, metrics
from photdis.stage1 import require_compute, structured, write_json

OUT = Path("data/followup_samefield_v1")
FIG = Path("figures/followup_samefield")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_inputs():
    config = json.loads(Path("configs/followup_fields.json").read_text())
    root = Path("data/cold_baseline_v3")
    expected = {"manifests/cold_baseline.json": config["baseline_manifest_sha256"],
                str(root / "test_predictions.fits"): config["baseline_test_predictions_sha256"]}
    expected.update({str(root / f"{name}.joblib"): value
                     for name, value in config["baseline_model_sha256"].items()})
    for name, digest in expected.items():
        if sha(name) != digest:
            raise ValueError(f"Frozen baseline input changed: {name}")
    return config, json.loads(Path("manifests/cold_baseline.json").read_text())


def ps1_masks(table):
    common = np.logical_and.reduce([table[f"ps1_{band}_quality"] & table[f"valid_{band}"]
                                    for band in "grz"])
    maximum = np.max(np.column_stack([np.abs(table[f"delta_{band}"]) for band in "grz"]), axis=1)
    return {"0.03": common & (maximum < .03), "0.05": common & (maximum < .05)}, maximum


def crossfit_strict(table, selected, part, cells, published, colors, mg):
    result = np.full((len(table), 5), np.nan)
    ids = table["source_id"]
    test = selected & (part == 2)
    assert identity_hash(ids[test]) == identity_hash(published["source_id"])
    lookup = {int(value): i for i, value in enumerate(published["source_id"])}
    for row in np.flatnonzero(test):
        i = lookup[int(ids[row])]
        result[row] = [published[f"A_grz_full_q{q}"][i]
                       for q in ("025", "160", "500", "840", "975")]
    pool = selected & (part != 2)
    unique = np.unique(cells[pool])
    shuffled = np.random.default_rng(20261030).permutation(unique)
    fold_map = {int(cell): i % 5 for i, cell in enumerate(shuffled)}
    fold = np.array([fold_map.get(int(cell), -1) for cell in cells])
    for i in range(5):
        fit = pool & (fold != i)
        held = pool & (fold == i)
        if not held.any():
            continue
        forest = QuantileForest(min_samples_leaf=10).fit(colors[fit], mg[fit])
        result[held] = forest.predict(colors[held])
    if not np.isfinite(result[selected]).all():
        raise ValueError("Missing out-of-fit strict prediction")
    return result, {"folds": 5, "unit": "nside64 NESTED cell",
                    "training_pool_excludes_frozen_test": True,
                    "frozen_test_prediction_rows": int(test.sum())}


def matched_residuals(table, recovery, control, residual):
    fields = ("gr0", "rz0", "gaia_phot_g_mean_mag", "ebv", "ra", "dec")
    scale = np.array([.15, .15, 1.0, .01, 5.0, 5.0])
    values = np.column_stack([table[name].astype(float) for name in fields])
    rec = np.flatnonzero(recovery)
    ctr = np.flatnonzero(control)
    if not len(rec) or not len(ctr):
        return {"n_pairs": 0}
    tree = cKDTree(values[ctr] / scale)
    distance, index = tree.query(values[rec] / scale, k=min(8, len(ctr)))
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    chosen = np.full(len(rec), -1, int)
    metric = np.full(len(rec), np.nan)
    for i in range(len(rec)):
        for j in range(index.shape[1]):
            candidate = ctr[index[i, j]]
            if np.all(np.abs(values[rec[i]] - values[candidate]) <= scale):
                chosen[i] = candidate
                metric[i] = distance[i, j]
                break
    good = chosen >= 0
    if not good.any():
        return {"n_pairs": 0}
    difference = residual[rec[good]] - residual[chosen[good]]
    return {"n_pairs": int(good.sum()), "unique_controls": int(len(np.unique(chosen[good]))),
            "matching_columns": list(fields), "maximum_absolute_differences": scale.tolist(),
            "matched_recovery_residual": stats(residual[rec[good]]),
            "matched_strict_residual": stats(residual[chosen[good]]),
            "paired_residual_difference": stats(difference),
            "standardized_distance": stats(metric[good])}


def replacement_result(table, use, model, legacy_pred, mg):
    if not use.any():
        return {"n": 0}, np.array([]), np.array([])
    ps1_colors, ps1_mg = ps1_replacement(table[use])
    ps1_pred = model.predict(ps1_colors)
    old = legacy_pred[use, 2] - mg[use]
    new = ps1_pred[:, 2] - ps1_mg
    result = {"n": int(use.sum()), "Legacy_residual": stats(old),
              "PS1_replacement_residual": stats(new),
              "Legacy_minus_PS1_residual": stats(old - new),
              "Legacy_minus_PS1_grz": {band: stats(table[f"delta_{band}"][use])
                                         for band in "grz"}}
    return result, old, new


def spectral_quality(table, test, predicted):
    calibrated, domain = rvs_metallicity(table["desi_feh"], table["desi_teff"], table["desi_logg"])
    masks = {"quality_DESI": test & table["desi_quality"],
             "quality_DESI_calibration_domain": test & table["desi_quality"] & domain}
    summary = {name: {"n": int(mask.sum()),
                      "raw_feh": stats(table["desi_feh"][mask]),
                      "calibrated_feh": stats(calibrated[mask]),
                      "residual": stats(predicted[mask, 2] -
                                        (table["g0"][mask] - 10 + 5 * np.log10(table["p_corr"][mask])))}
               for name, mask in masks.items()}
    return summary


def main():
    require_compute()
    if (OUT / "summary.json").exists():
        raise FileExistsError(OUT / "summary.json")
    config, baseline = frozen_inputs()
    table = fitsio.read("data/pilot_audit_v1/reference.fits")
    published = fitsio.read("data/cold_baseline_v3/test_predictions.fits")
    assert len(table) == 6942 and len(published) == 238
    holdout = annotate_holdouts(load_reference_catalogs("data/reference"),
                                table["source_id"], table["gaia_ra"], table["gaia_dec"])
    eligible, flags, mg = common_nonoptical(table, holdout["external_any"])
    strict = table["anchor"] & eligible
    if int(strict.sum()) != baseline["selected_count"]:
        raise ValueError("Strict cold-dwarf identity changed")
    cells = hp.ang2pix(64, table["gaia_ra"], table["gaia_dec"], nest=True, lonlat=True)
    map_frozen = baseline["split"]["cell_map"]
    part = np.array([map_frozen.get(str(int(c)), 3) for c in cells], np.int8)
    if not np.array_equal(np.sort(table["source_id"][strict & (part == 2)]),
                          np.sort(published["source_id"])):
        raise ValueError("Frozen test IDs changed")
    colors = np.column_stack((table["gr0"], table["rz0"]))
    model = joblib.load("data/cold_baseline_v3/A_grz.joblib")
    strict_pred, crossfit = crossfit_strict(table, strict, part, cells, published, colors, mg)
    ps1, max_delta = ps1_masks(table)
    candidate = np.isin(table["audit_group"], [1, 2, 3])
    old = (candidate & ps1["0.05"] & ~holdout["external_any"] &
           table["bp_rp_precise"] & (table["bp_rp_checked"] >= 1.3) &
           (table["bp_rp_checked"] < 3.5) & (mg > 5) & table["grz_base"])
    if int(old.sum()) != baseline["recovered_ps1_diagnostic"]["n"]:
        raise ValueError("Published 324-star recovery sample changed")
    steps = []
    current = old.copy()
    for name, flag in flags.items():
        before = int(current.sum())
        current &= flag
        steps.append({"condition": name, "before": before,
                      "removed": before - int(current.sum()), "remaining": int(current.sum())})
    recovered = candidate & ps1["0.05"] & eligible
    assert np.array_equal(current, recovered)
    recovered03 = candidate & ps1["0.03"] & eligible
    recovered_pred = np.full_like(strict_pred, np.nan)
    recovered_pred[recovered] = model.predict(colors[recovered])
    residual = np.where(recovered, recovered_pred[:, 2] - mg, strict_pred[:, 2] - mg)
    strict_control = strict & ps1["0.05"]
    test_control = strict_control & (part == 2)
    if not test_control.any():
        raise ValueError("No frozen strict test stars share the PS1 comparison conditions")
    summary = {"baseline_commit": config["baseline_commit"],
               "frozen_test_id_sha256": identity_hash(published["source_id"]),
               "strict_cold": int(strict.sum()), "strict_ps1_control": int(strict_control.sum()),
               "strict_ps1_frozen_test_control": int(test_control.sum()),
               "original_recovery": int(old.sum()), "eligibility_flow": steps,
               "recovery_0p03": int(recovered03.sum()),
               "recovery_0p05": int(recovered.sum()),
               "recovery_groups": {str(j): int((recovered & (table["audit_group"] == j)).sum())
                                   for j in (1, 2, 3)},
               "frozen_model_recovery": metrics(mg[recovered], recovered_pred[recovered],
                                                table["g0"][recovered], table["p_corr"][recovered]),
               "strict_test_comparison": metrics(mg[test_control], strict_pred[test_control]),
               "crossfit": crossfit,
               "matched_test_controls": matched_residuals(table, recovered, test_control, residual),
               "matched_all_out_of_fit_controls": matched_residuals(table, recovered, strict_control, residual),
               "paired_replacement": {},
               "spectral_quality_test": spectral_quality(table, strict & (part == 2), strict_pred),
               "compute_host": socket.gethostname()}
    paired = {}
    for name, mask in (("recovery_0p03", recovered03), ("recovery_0p05", recovered),
                       ("strict_frozen_test", test_control)):
        prediction = recovered_pred if name.startswith("recovery") else strict_pred
        info, old_residual, ps1_residual = replacement_result(table, mask, model, prediction, mg)
        summary["paired_replacement"][name] = info
        paired[name] = (mask, old_residual, ps1_residual)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(OUT / "recovery_rows.fits"), structured({
        "source_id": table["source_id"][recovered], "audit_group": table["audit_group"][recovered],
        "gr0": table["gr0"][recovered], "rz0": table["rz0"][recovered],
        "gaia_G": table["gaia_phot_g_mean_mag"][recovered],
        "mg_gaia": mg[recovered], "residual_legacy": residual[recovered],
        "maximum_ps1_delta": max_delta[recovered]}), extname="RECOVERY", clobber=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, color in (("recovery_0p05", "#c5544b"), ("strict_frozen_test", "#437ca6")):
        mask, old_residual, ps1_residual = paired[name]
        axes[0].scatter(table["bp_rp_checked"][mask], old_residual, s=12, alpha=.55,
                        label=f"{name}: {mask.sum()}", color=color)
        axes[1].hist(old_residual - ps1_residual, bins=np.linspace(-.4, .4, 51),
                     histtype="step", label=name, color=color)
    axes[0].set(xlabel="Gaia BP-RP", ylabel="Legacy model residual [mag]", ylim=(-1.5, 1.5))
    axes[1].set(xlabel="Residual Legacy - residual PS1 replacement [mag]", ylabel="Stars")
    axes[0].axhline(0, color=".4")
    axes[1].axvline(0, color=".4")
    for axis in axes: axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "01_recovery_and_photometry.png", dpi=170); plt.close(fig)
    write_json(OUT / "summary.json", summary)
    write_json("manifests/followup_samefield.json", summary)
    print(json.dumps({key: summary[key] for key in
                      ("original_recovery", "eligibility_flow", "recovery_0p03", "recovery_0p05",
                       "frozen_model_recovery", "matched_test_controls", "paired_replacement",
                       "spectral_quality_test")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

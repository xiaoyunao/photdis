#!/usr/bin/env python3
"""Compare one selected XD candidate with frozen-v4 on identical development stars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import joblib
import numpy as np

from photdis.route import FEATURES, LEVEL, feature_matrix, route_rows
from photdis.route_audit import higher_tier_reasons
from photdis.route_policy import choose_policy
from photdis.stage1 import require_compute, write_json
from photdis.xd_distance import measurement_covariance
from cold_v4 import OUT as V4_OUT, PREPARED, PREPARED_TRACKED, sha
from cold_v5_xd_fit import BANDS, CONFIG, add_sidecar, load_development
from photdis.holdouts import load_reference_catalogs

LOCK = Path("configs/cold_v4_deep_policy.json")
SELECTION = Path("manifests/cold_v5_xd_selection.json")
OUTPUT = Path("data/cold_v5_xd/comparison.json")
SMALL = Path("manifests/cold_v5_xd_comparison.json")


def statistics(truth, prediction):
    if not len(truth):
        return {"n": 0}
    residual = prediction[:, 2] - truth
    center = np.median(residual)
    return {"n": len(truth), "center_mag": float(center),
            "nmad_mag": float(1.4826*np.median(np.abs(residual-center))),
            "abs_gt_0p5": float(np.mean(np.abs(residual) > .5)),
            "coverage68": float(np.mean((truth >= prediction[:, 1]) &
                                         (truth <= prediction[:, 3]))),
            "coverage95": float(np.mean((truth >= prediction[:, 0]) &
                                         (truth <= prediction[:, 4]))),
            "below95": float(np.mean(truth < prediction[:, 0])),
            "above95": float(np.mean(truth > prediction[:, 4])),
            "median_width95_mag": float(np.median(prediction[:, 4]-prediction[:, 0]))}


def response_stats(change):
    return {"n": len(change), "median_mag": float(np.median(change)),
            "median_abs_mag": float(np.median(np.abs(change))),
            "nmad_mag": float(1.4826*np.median(np.abs(change-np.median(change)))),
            "fraction_abs_gt_0p05": float(np.mean(np.abs(change) > .05))}


def perturbed_features(table, tier, use, band, delta):
    color = feature_matrix(table, tier)[use].copy()
    g = np.asarray(table["g0"][use], float).copy()
    if band == "g":
        g += delta
        color[:, 0] += delta
    elif band == "r":
        color[:, 0] -= delta
        color[:, 1] += delta
    else:
        raise ValueError(band)
    return g, color


def field_compare(source, route, policy, forests, mixtures):
    table, cold, mg, chosen, meta = source
    selected = cold & (chosen > 0)
    routed = route_rows(table, route)
    reasons = higher_tier_reasons(table, routed, chosen)
    truth = 10 - 5*np.log10(np.asarray(table["p_corr"][selected], float))
    qrf = np.empty((selected.sum(), 5))
    xd = np.empty_like(qrf)
    optical = np.asarray(table["gr0"][selected], float)
    used_error = np.empty(selected.sum(), float)
    tier_selected = chosen[selected]
    conflict = reasons["higher_support_conflict"][selected]
    perturb = {"g": {"qrf": np.empty(selected.sum()), "xd": np.empty(selected.sum())},
               "r": {"qrf": np.empty(selected.sum()), "xd": np.empty(selected.sum())}}
    for tier, level in LEVEL.items():
        local = tier_selected == level
        if not local.any():
            continue
        use = selected & (chosen == level)
        colors = feature_matrix(table, tier)[use]
        g0 = np.asarray(table["g0"][use], float)
        sigma = np.column_stack([table[f"sigma_{band}"][use] for band in BANDS[tier]])
        cov = measurement_covariance(sigma)
        v4mg = forests[tier].predict(colors)
        qrf[local] = g0[:, None] - v4mg[:, ::-1]
        xd[local] = mixtures[tier].posterior_mu(np.column_stack([g0, colors]), cov)
        used_error[local] = np.max(sigma, axis=1)
        for band in perturb:
            gp, cp = perturbed_features(table, tier, use, band, .01)
            v4_shift = gp - forests[tier].predict(cp)[:, 2]
            xd_shift = mixtures[tier].posterior_mu(np.column_stack([gp, cp]), cov,
                                                   quantiles=(.5,))[:, 0]
            perturb[band]["qrf"][local] = v4_shift-qrf[local, 2]
            perturb[band]["xd"][local] = xd_shift-xd[local, 2]
    field = {"field": meta["name"], "n_same_selected": int(selected.sum()),
             "tier_counts": {tier: int(np.sum(tier_selected == level))
                             for tier, level in LEVEL.items()},
             "infrared_fallback_conflict": int(conflict.sum()),
             "qrf": statistics(truth, qrf), "xd": statistics(truth, xd),
             "by_tier": {}, "by_error_tertile": {}, "by_color": {},
             "by_conflict": {}, "perturbation_plus_0p01_mag": {}}
    for tier, level in LEVEL.items():
        mask = tier_selected == level
        field["by_tier"][tier] = {"qrf": statistics(truth[mask], qrf[mask]),
                                  "xd": statistics(truth[mask], xd[mask])}
    edges = np.quantile(used_error, [0, 1/3, 2/3, 1])
    field["error_tertile_edges"] = edges.tolist()
    for j in range(3):
        mask = (used_error >= edges[j]) & ((used_error < edges[j+1]) if j < 2
                                               else (used_error <= edges[j+1]))
        field["by_error_tertile"][str(j)] = {
            "qrf": statistics(truth[mask], qrf[mask]),
            "xd": statistics(truth[mask], xd[mask])}
    color_edges = (.9, 1.2, 1.6, 2.0, 4.0)
    for lo, hi in zip(color_edges[:-1], color_edges[1:]):
        mask = (optical >= lo) & (optical < hi)
        field["by_color"][f"{lo:g}_{hi:g}"] = {
            "qrf": statistics(truth[mask], qrf[mask]),
            "xd": statistics(truth[mask], xd[mask])}
    for name, mask in (("conflict", conflict), ("no_conflict", ~conflict)):
        field["by_conflict"][name] = {"qrf": statistics(truth[mask], qrf[mask]),
                                      "xd": statistics(truth[mask], xd[mask])}
    for band in perturb:
        field["perturbation_plus_0p01_mag"][band] = {
            model: response_stats(change) for model, change in perturb[band].items()}
    return field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, required=True)
    args = parser.parse_args()
    require_compute()
    if OUTPUT.exists() or SMALL.exists():
        raise FileExistsError("Comparison already exists")
    cfg = json.loads(CONFIG.read_text())
    selected_size = json.loads(SELECTION.read_text())
    if args.components != selected_size["selected_components"] or sha(CONFIG) != selected_size["configuration_sha256"]:
        raise ValueError("Selected XD size/configuration changed")
    if sha(PREPARED) != cfg["v4_prepared_sha256"] or sha(PREPARED) != sha(PREPARED_TRACKED):
        raise ValueError("Frozen v4 preparation changed")
    lock = json.loads(LOCK.read_text())
    if lock["chosen_policy"] != cfg["v4_policy"]:
        raise ValueError("WISE20 policy changed")
    prepared = json.loads(PREPARED.read_text())
    route_path = V4_OUT / "route.joblib"
    if sha(route_path) != prepared["route"]["route_sha256"]:
        raise ValueError("Frozen route changed")
    route = joblib.load(route_path)
    fit = json.loads((Path("manifests") / f"cold_v5_xd_k{args.components:02d}.json").read_text())
    forests, mixtures = {}, {}
    for tier in FEATURES:
        forest_path = V4_OUT / f"{tier}.joblib"
        mixture_path = Path("data/cold_v5_xd") / f"k{args.components:02d}" / f"{tier}.joblib"
        if sha(forest_path) != prepared["tier_training"][tier]["model_sha256"] or (
                sha(mixture_path) != fit["tiers"][tier]["model_sha256"]):
            raise ValueError(f"Frozen {tier} model changed")
        forests[tier] = joblib.load(forest_path)
        mixtures[tier] = joblib.load(mixture_path)
    references = load_reference_catalogs("data/reference")
    paths = [*cfg["configuration_development_fields"], cfg["small_diagnostic_field"]]
    report = {"components": args.components, "configuration_sha256": sha(CONFIG),
              "selection_manifest_sha256": sha(SELECTION),
              "v4_prepared_sha256": sha(PREPARED),
              "policy": cfg["v4_policy"],
              "distance_modulus_prior": cfg["distance_modulus_prior"],
              "fields": {}, "compute_host": socket.gethostname()}
    for path in paths:
        source = load_development(path, references, route, lock["chosen_policy"])
        field = field_compare(source, route, lock["chosen_policy"], forests, mixtures)
        report["fields"][field["field"]] = field
        print(json.dumps({"field": field["field"], "n": field["n_same_selected"],
                          "qrf_nmad": field["qrf"].get("nmad_mag"),
                          "xd_nmad": field["xd"].get("nmad_mag")}), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, report)
    write_json(SMALL, report)


if __name__ == "__main__":
    main()

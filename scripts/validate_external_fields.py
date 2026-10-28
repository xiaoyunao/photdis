#!/usr/bin/env python3
"""Direct transfer and preassigned-field CQR of the frozen pilot forest."""
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

from photdis.audit import checked_gaia_color, stats
from photdis.followup import apply_cqr, common_nonoptical, cqr_offset, identity_hash
from photdis.holdouts import annotate_holdouts, load_reference_catalogs
from photdis.quantile import metrics
from photdis.stage1 import require_compute, structured, write_json

OUT = Path("data/external_validation_v1")
FIG = Path("figures/external_validation")
FEATURES = {"A_grz": ("gr0", "rz0"),
            "B_grzW1": ("gr0", "rz0", "zw10"),
            "C_grzW1W2": ("gr0", "rz0", "zw10", "w1w20")}
TIERS = {"A_grz": "grz_base", "B_grzW1": "grz_w1", "C_grzW1W2": "grz_w1_w2"}
GAIA_COLOR_FIELDS = ["gaia_gaia_match", "gaia_phot_g_mean_mag", "gaia_phot_bp_mean_mag", "gaia_phot_rp_mean_mag",
                     "gaia_phot_bp_mean_flux", "gaia_phot_rp_mean_flux",
                     "gaia_phot_bp_mean_flux_error", "gaia_phot_rp_mean_flux_error",
                     "gaia_phot_bp_n_obs", "gaia_phot_rp_n_obs", "gaia_bp_rp"]
FIELDS = ["source_id", "ra", "dec", "ebv", "g0", "p_corr", "snr_corr", "sigma_g",
          "anchor", "anchor_astrometry", "contaminant_candidate",
          "desi_quality", "desi_known_contaminant", "desi_match_ambiguous", "desi_logg",
          "apogee_quality", "apogee_known_contaminant", "apogee_match_ambiguous", "apogee_logg",
          "grz_base", "grz_w1", "grz_w1_w2", "gr0", "rz0", "zw10", "w1w20"] + GAIA_COLOR_FIELDS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_field(field, references):
    name = field["name"]
    # Resolve the registered field name to its pilot output directory.
    matches = [Path("configs/field_cal_ra140_150.json"),
               Path("configs/field_test_ra190_200.json"),
               Path("configs/field_test_ra320_330.json")]
    tasks = [json.loads(p.read_text()) for p in matches]
    task = next(item for item in tasks if item["name"] == name)
    path = Path(task["output_dir"]) / "field_diagnostics.fits"
    with fitsio.FITS(str(path)) as file:
        rows = np.flatnonzero(file[1].read(columns=["anchor"])["anchor"])
        raw = file[1].read(columns=FIELDS, rows=rows)
    table = {key: raw[key] for key in raw.dtype.names}
    gaia = {key[5:]: table[key] for key in GAIA_COLOR_FIELDS}
    table.update(checked_gaia_color(gaia))
    external = annotate_holdouts(references, table["source_id"],
                                 table["ra"], table["dec"])["external_any"]
    eligible, _, mg = common_nonoptical(table, external)
    selected = table["anchor"] & eligible
    if not selected.any():
        raise ValueError(f"No cold strict anchors in {name}")
    meta = {"role": field["role"], "bounds_deg": field["bounds_deg"],
            "all_strict_anchor_rows": len(raw), "cold_strict_anchor_rows": int(selected.sum()),
            "external_anchor_rows": int(external.sum()),
            "cold_identity_sha256": identity_hash(table["source_id"][selected]),
            "Gaia_G": stats(table["gaia_phot_g_mean_mag"][selected]),
            "BP_RP": stats(table["bp_rp_checked"][selected]),
            "EBV": stats(table["ebv"][selected])}
    return table, selected, mg, meta


def tier_prediction(table, selected, mg, name, model):
    mask = selected & table[TIERS[name]]
    colors = np.column_stack([table[key][mask].astype(float) for key in FEATURES[name]])
    good = np.all(np.isfinite(colors), axis=1)
    rows = np.flatnonzero(mask)[good]
    colors = colors[good]
    mask = np.zeros(len(selected), bool)
    mask[rows] = True
    if len(rows) == 0:
        raise ValueError(f"No valid {name} anchors in external field")
    prediction = model.predict(colors)
    truth = mg[mask]
    direct = metrics(truth, prediction, table["g0"][mask], table["p_corr"][mask])
    result = {"mask": mask, "prediction": prediction, "truth": truth,
              "direct": direct, "ids": table["source_id"][mask],
              "g0": table["g0"][mask], "p_corr": table["p_corr"][mask],
              "color": table["bp_rp_checked"][mask]}
    return result


def red_bins(result, adjusted):
    bins = [1.3, 1.8, 2.3, 2.8, 3.5]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        take = (result["color"] >= lo) & (result["color"] < hi)
        if take.sum() < 5:
            rows.append({"lo": lo, "hi": hi, "n": int(take.sum())})
        else:
            rows.append({"lo": lo, "hi": hi,
                         "direct": metrics(result["truth"][take], result["prediction"][take]),
                         "calibrated": metrics(result["truth"][take], adjusted[take])})
    return rows


def main():
    require_compute()
    if (OUT / "summary.json").exists():
        raise FileExistsError(OUT / "summary.json")
    config = json.loads(Path("configs/followup_fields.json").read_text())
    if sha("manifests/cold_baseline.json") != config["baseline_manifest_sha256"]:
        raise ValueError("Published baseline manifest changed")
    if sha("data/cold_baseline_v3/test_predictions.fits") != config["baseline_test_predictions_sha256"]:
        raise ValueError("Frozen pilot test identities changed")
    models = {}
    for name, expected in config["baseline_model_sha256"].items():
        path = Path("data/cold_baseline_v3") / f"{name}.joblib"
        if sha(path) != expected:
            raise ValueError(f"Frozen model changed: {name}")
        models[name] = joblib.load(path)
    refs = load_reference_catalogs("data/reference")
    fields = {}
    summary = {"baseline_commit": config["baseline_commit"],
               "preassigned_roles": {item["name"]: item["role"] for item in config["fields"]},
               "calibration_method": "split CQR; max(q_lower-y,y-q_upper); rank ceil((n+1)(1-alpha)); nonnegative expansion",
               "fields": {}, "compute_host": socket.gethostname()}
    for field in config["fields"]:
        table, selected, mg, meta = load_field(field, refs)
        tiers = {name: tier_prediction(table, selected, mg, name, model)
                 for name, model in models.items()}
        fields[field["name"]] = tiers
        summary["fields"][field["name"]] = {**meta,
            "tiers": {name: {"n": result["direct"]["n"], "direct": result["direct"],
                             "raw_width_68_median": float(np.median(result["prediction"][:, 3] - result["prediction"][:, 1])),
                             "raw_width_95_median": float(np.median(result["prediction"][:, 4] - result["prediction"][:, 0]))}
                      for name, result in tiers.items()}}
        print(json.dumps({"phase": "direct field transfer", "field": field["name"],
                          "cold": int(selected.sum()),
                          "tiers": {name: value["direct"] for name, value in tiers.items()}}), flush=True)
    calibration_name = next(field["name"] for field in config["fields"]
                            if field["role"] == "external_calibration")
    offsets = {}
    for name, result in fields[calibration_name].items():
        q68, n68 = cqr_offset(result["truth"], result["prediction"], 1, 3, .32)
        q95, n95 = cqr_offset(result["truth"], result["prediction"], 0, 4, .05)
        offsets[name] = {"q68": q68, "q95": q95, "n": n68}
        adjusted = apply_cqr(result["prediction"], q68, q95)
        summary["fields"][calibration_name]["tiers"][name]["calibration_fit"] = metrics(result["truth"], adjusted)
    summary["offsets_mag"] = offsets
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for field in config["fields"]:
        if field["role"] != "external_final_test":
            continue
        name = field["name"]
        for tier, result in fields[name].items():
            adjusted = apply_cqr(result["prediction"], offsets[tier]["q68"], offsets[tier]["q95"])
            item = summary["fields"][name]["tiers"][tier]
            item["calibrated"] = metrics(result["truth"], adjusted, result["g0"], result["p_corr"])
            item["calibrated_width_68_median"] = float(np.median(adjusted[:, 3] - adjusted[:, 1]))
            item["calibrated_width_95_median"] = float(np.median(adjusted[:, 4] - adjusted[:, 0]))
            item["color_bins"] = red_bins(result, adjusted)
            write = {"source_id": result["ids"], "mg_gaia": result["truth"],
                     "bp_rp": result["color"], "g0": result["g0"]}
            for i, quantile in enumerate(("025", "160", "500", "840", "975")):
                write[f"raw_q{quantile}"] = result["prediction"][:, i]
                write[f"cal_q{quantile}"] = adjusted[:, i]
            fitsio.write(str(OUT / f"{name}_{tier}.fits"), structured(write),
                         extname="EXTERNAL_TEST", clobber=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    final = [item for item in config["fields"] if item["role"] == "external_final_test"]
    labels = [item["name"].replace("test_", "") for item in final]
    x = np.arange(len(final))
    for i, tier in enumerate(FEATURES):
        bias = [summary["fields"][item["name"]]["tiers"][tier]["direct"]["bias_mag"] for item in final]
        raw = [summary["fields"][item["name"]]["tiers"][tier]["direct"]["coverage_95"] for item in final]
        cal = [summary["fields"][item["name"]]["tiers"][tier]["calibrated"]["coverage_95"] for item in final]
        axes[0].scatter(x + (i-1)*.12, bias, label=tier, s=55)
        axes[1].scatter(x + (i-1)*.12, raw, marker="o", label=f"{tier} raw", s=48)
        axes[1].scatter(x + (i-1)*.12, cal, marker="x", label=f"{tier} CQR", s=48)
    axes[0].axhline(0, color=".5", ls=":")
    axes[1].axhline(.95, color=".5", ls=":")
    axes[0].set(ylabel="Predicted - Gaia $M_g$ median [mag]", xticks=x, xticklabels=labels)
    axes[1].set(ylabel="Actual 95% coverage", xticks=x, xticklabels=labels, ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Frozen pilot model in preassigned external final-test fields")
    fig.tight_layout(); fig.savefig(FIG / "01_external_transfer_and_calibration.png", dpi=170); plt.close(fig)
    write_json(OUT / "summary.json", summary)
    write_json("manifests/external_validation.json", summary)
    print(json.dumps({"offsets": offsets,
                      "final_tests": {item["name"]: summary["fields"][item["name"]]["tiers"]
                                      for item in final}}, indent=2), flush=True)


if __name__ == "__main__":
    main()

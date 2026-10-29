#!/usr/bin/env python3
"""Fixed low-degree cluster RGB transfer diagnostic; NGC 7006 is test-only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps"), str(ROOT / "scripts")]
import fitsio
import numpy as np

from cluster_rgb_phase import CONFIG, classify
from photdis.stage1 import require_compute, write_json

DEVELOPMENT = {"m3": ("lower_rgb",),
               "m53": ("upper_rgb", "lower_rgb"),
               "ngc2419": ("upper_rgb",)}
HOLDOUT = "ngc7006"
DUST_SIGMA = {"m3": .005, "m53": .010, "ngc2419": .020,
              "ngc7006": .010}
RG, RR, RZ = 3.214, 2.165, 1.211
RIDGE = .001
MODEL_PATH = Path("manifests/cluster_rgb_transfer_model_v1.json")
TEST_PATH = Path("manifests/cluster_rgb_transfer_holdout_v1.json")


def load_cluster(name, regions):
    tab = fitsio.read(str(Path("data") / f"cluster_rgb_{name}_v1" / "members.fits"))
    inv = json.loads((Path("manifests") / f"cluster_rgb_{name}_inventory_v1.json").read_text())
    cfg = json.loads(CONFIG.read_text())["systems"][name]
    mu = inv["external_modulus"]["mu_mag"]
    phase = classify(tab["gaia_g"]-mu, tab["bp_rp"], cfg)
    mask = ((tab["pmember"] > .9) & tab["grz_clean"]
            & np.isin(phase, regions))
    for col in ("g0", "gr0", "rz0"):
        mask &= np.isfinite(tab[col])
    selected = tab[mask]
    if not len(selected):
        raise ValueError(f"No selected RGB members in {name}")
    return {"name": name, "id": np.asarray(selected["source_id"], np.int64),
            "g0": np.asarray(selected["g0"], float),
            "gr0": np.asarray(selected["gr0"], float),
            "rz0": np.asarray(selected["rz0"], float),
            "mg": np.asarray(selected["g0"], float)-mu,
            "phase": phase[mask],
            "w1": np.asarray(selected["w1_clean"], bool),
            "w2": np.asarray(selected["w2_clean"], bool),
            "mu": mu, "mu_stat": inv["external_modulus"]["stat_sigma_mag"],
            "mu_sys": inv["external_modulus"]["systematic_sigma_mag"]}


def colors(data, variant):
    first = np.asarray(data["gr0"], float)
    if variant == "A":
        return first[:, None]
    if variant == "B":
        return np.column_stack((first, np.asarray(data["rz0"], float)))
    raise ValueError(variant)


def matrix(z, variant):
    x = z[:, 0]
    if variant == "A":
        return np.column_stack((np.ones(len(x)), x, x*x))
    y = z[:, 1]
    return np.column_stack((np.ones(len(x)), x, y, x*x, x*y, y*y))


def fit_model(groups, variant):
    raw = np.concatenate([colors(group, variant) for group in groups.values()])
    mg = np.concatenate([group["mg"] for group in groups.values()])
    weight = np.concatenate([np.full(len(group["mg"]), 1/(len(groups)*len(group["mg"])))
                             for group in groups.values()])
    mean = np.average(raw, weights=weight, axis=0)
    scale = np.sqrt(np.average((raw-mean)**2, weights=weight, axis=0))
    if np.any(scale <= 0):
        raise ValueError("Degenerate RGB color scale")
    x = matrix((raw-mean)/scale, variant)
    penalty = np.diag([0.] + [RIDGE]*(x.shape[1]-1))
    beta = np.linalg.solve(x.T @ (weight[:, None]*x)+penalty, x.T @ (weight*mg))
    return {"variant": variant, "coefficients": beta.tolist(),
            "color_mean": mean.tolist(), "color_scale": scale.tolist(),
            "color_p01": np.percentile(raw, 1, axis=0).tolist(),
            "color_p99": np.percentile(raw, 99, axis=0).tolist(),
            "ridge": RIDGE, "total_weight_per_system": 1/len(groups),
            "training_systems": list(groups)}


def predict(model, group):
    variant = model["variant"]
    raw = colors(group, variant)
    mean = np.asarray(model["color_mean"])
    scale = np.asarray(model["color_scale"])
    z = (raw-mean)/scale
    return matrix(z, variant) @ np.asarray(model["coefficients"])


def support(model, group):
    raw = colors(group, model["variant"])
    return ((raw >= model["color_p01"]) &
            (raw <= model["color_p99"])).all(axis=1)


def score(model, group, mask=None):
    if mask is None:
        mask = np.ones(len(group["mg"]), bool)
    residual = group["mg"][mask]-predict(model, group)[mask]
    if not len(residual):
        return {"n": 0}
    center = np.median(residual)
    return {"n": int(len(residual)), "delta_mu_median": float(center),
            "delta_mu_nmad": float(1.4826*np.median(np.abs(residual-center))),
            "abs_delta_mu_gt_0p5_fraction": float(np.mean(np.abs(residual) > .5))}


def shifted(group, *, dmu=0., debv=0.):
    out = dict(group)
    out["g0"] = group["g0"]-RG*debv
    out["gr0"] = group["gr0"]-(RG-RR)*debv
    out["rz0"] = group["rz0"]-(RR-RZ)*debv
    out["mg"] = out["g0"]-(group["mu"]+dmu)
    return out


def groups_hash(groups):
    digest = hashlib.sha256()
    for name, group in groups.items():
        digest.update(name.encode())
        digest.update(np.sort(group["id"]).astype("<i8").tobytes())
    return digest.hexdigest()


def train():
    if MODEL_PATH.exists():
        raise FileExistsError(MODEL_PATH)
    groups = {name: load_cluster(name, regions)
              for name, regions in DEVELOPMENT.items()}
    models = {v: fit_model(groups, v) for v in ("A", "B")}
    summary = {v: {name: score(model, group) for name, group in groups.items()}
               for v, model in models.items()}
    loso = {v: {name: score(fit_model({other: data for other, data in groups.items()
                                       if other != name}, v), group)
                for name, group in groups.items()} for v in ("A", "B")}
    out = {"protocol": "docs/cluster_rgb_experiment_protocol_v1.md",
           "development_regions": DEVELOPMENT, "holdout_system": HOLDOUT,
           "source_identity_sha256": groups_hash(groups),
           "counts": {name: {"n": len(group["id"]),
                              "w1_clean": int(group["w1"].sum()),
                              "w1w2_clean": int((group["w1"] & group["w2"]).sum())}
                      for name, group in groups.items()},
           "models": models, "development_scores": summary,
           "leave_one_system_out_scores": loso,
           "compute_host": socket.gethostname()}
    write_json(MODEL_PATH, out)
    print(json.dumps({"counts": out["counts"], "development_scores": summary,
                      "leave_one_system_out_scores": loso}), flush=True)


def bootstrap_median(residual, seed=20261029):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(residual), (2000, len(residual)))
    return np.percentile(np.median(residual[idx], axis=1), [2.5, 97.5]).tolist()


def test():
    if TEST_PATH.exists():
        raise FileExistsError(TEST_PATH)
    frozen = json.loads(MODEL_PATH.read_text())
    groups = {name: load_cluster(name, regions)
              for name, regions in DEVELOPMENT.items()}
    if groups_hash(groups) != frozen["source_identity_sha256"]:
        raise ValueError("Development membership differs from frozen model")
    hold = load_cluster(HOLDOUT, ("upper_rgb",))
    lower = load_cluster(HOLDOUT, ("lower_rgb",))
    results = {}
    sensitivity = {}
    for variant, model in frozen["models"].items():
        keep = support(model, hold)
        residual = hold["mg"]-predict(model, hold)
        results[variant] = {"whole_upper_rgb": score(model, hold),
                            "in_training_color_rectangle": score(model, hold, keep),
                            "color_rectangle_n": int(keep.sum()),
                            "source_bootstrap_median_95pct": bootstrap_median(residual),
                            "lower_rgb_sensitivity": score(model, lower),
                            "w1_clean_upper": int(hold["w1"].sum()),
                            "w1w2_clean_upper": int((hold["w1"] & hold["w2"]).sum())}
        shared = {}
        baseline = float(np.median(residual))
        for name, group in groups.items():
            mu_sigma = float(np.hypot(group["mu_stat"], group["mu_sys"]))
            for shift_name, amount, kind in (("mu", mu_sigma, "dmu"),
                                             ("dust", DUST_SIGMA[name], "debv")):
                deltas = []
                for sign in (-1, 1):
                    perturbed = dict(groups)
                    perturbed[name] = shifted(group, **{kind: sign*amount})
                    refit = fit_model(perturbed, variant)
                    deltas.append(float(np.median(hold["mg"]-predict(refit, hold))-baseline))
                shared[f"{name}_{shift_name}"] = {"input_pm": amount,
                                                   "holdout_median_delta_pm": deltas}
        dust_hold = []
        for sign in (-1, 1):
            perturbed_hold = shifted(hold, debv=sign*DUST_SIGMA[HOLDOUT])
            dust_hold.append(float(np.median(perturbed_hold["mg"]-
                                             predict(model, perturbed_hold))-baseline))
        sensitivity[variant] = {"development_shared_shifts": shared,
                                "holdout_dust_ebv_pm": DUST_SIGMA[HOLDOUT],
                                "holdout_median_delta_dust_pm": dust_hold,
                                "holdout_modulus_sigma_mag": float(np.hypot(
                                    hold["mu_stat"], hold["mu_sys"]))}
    out = {"protocol": frozen["protocol"], "frozen_model": str(MODEL_PATH),
           "frozen_source_identity_sha256": frozen["source_identity_sha256"],
           "holdout_system": HOLDOUT, "primary_region": "upper_rgb",
           "secondary_region": "lower_rgb", "results": results,
           "shared_scale_sensitivity": sensitivity,
           "compute_host": socket.gethostname()}
    write_json(TEST_PATH, out)
    print(json.dumps({"results": results, "shared_scale_sensitivity": sensitivity}), flush=True)


def main():
    require_compute()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "test"))
    args = parser.parse_args()
    if args.mode == "train":
        train()
    else:
        test()


if __name__ == "__main__":
    main()

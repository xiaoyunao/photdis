#!/usr/bin/env python3
"""Fit registered DECam RGB forward models using development clusters only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.cluster_rgb_conditional import (RG, RR, RZ, fit_gls, support_bins)
from photdis.stage1 import require_compute, write_json

CONFIG = Path("configs/cluster_rgb_conditional_v1.json")
OUTPUT = Path("manifests/cluster_rgb_conditional_model_v1.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_training(config, tier):
    parts = []
    for name, settings in config["development"].items():
        tab = fitsio.read(f"data/cluster_rgb_gr_v2/{name}.fits")
        phase = tab["phase"].astype("U")
        mask = (tab["pmember"] > .9) & np.isin(phase, settings["phases"])
        mask &= tab["gr_clean"] if tier == "gr_all" else tab["grz_clean_same_row"]
        need = ["g0", "gr0", "sigma_g", "sigma_r"]
        if tier == "grz_common":
            need += ["rz0", "sigma_z"]
        mask &= np.isfinite(sum(tab[k].astype(float) for k in need))
        selected = tab[mask]
        if not len(selected):
            raise ValueError(f"No eligible {name} {tier}")
        parts.append({"name":name, "tab":selected,
                      "stage":phase[mask],
                      "mg": selected["g0"].astype(float)-settings.get(
                          "mu", {"m3":15.041,"m53":16.403,"ngc7006":18.07448608}[name]),
                      "feh":settings["feh_uves_rrab"],
                      "feh_sigma":settings["feh_sigma"],
                      "mu_sigma":{"m3":np.hypot(.017,.036),
                                  "m53":np.hypot(.024,.033),
                                  "ngc7006":.07378790}[name],
                      "dust_sigma":settings["dust_scale_sigma_ebv"]})
    ids = [set(p["tab"]["source_id"].astype(int)) for p in parts]
    if any(ids[i] & ids[j] for i in range(len(ids)) for j in range(i+1,len(ids))):
        raise ValueError("Development source identity overlap")
    return parts


def joined(parts, rng, shared_zero_sigma, *, perturb):
    arrays = {k:[] for k in ("mg","feh","color","sg","sr","sz","cluster","stage")}
    for p in parts:
        tab, n = p["tab"], len(p["tab"])
        if perturb:
            dmu = rng.normal(0,p["mu_sigma"])
            dust = rng.normal(0,p["dust_sigma"])
            zg,zr,zz = rng.normal(0,shared_zero_sigma,3)
            feh = rng.normal(p["feh"],p["feh_sigma"])
        else:
            dmu = dust = zg = zr = zz = 0.0
            feh = p["feh"]
        arrays["mg"].append(p["mg"]-dmu-RG*dust+zg)
        arrays["feh"].append(np.full(n,feh))
        arrays["color"].append(np.column_stack((
            tab["gr0"].astype(float)-(RG-RR)*dust+zg-zr,
            tab["rz0"].astype(float)-(RR-RZ)*dust+zr-zz
            if "rz0" in tab.dtype.names and np.all(np.isfinite(tab["rz0"]))
            else np.full(n,np.nan))))
        arrays["sg"].append(tab["sigma_g"].astype(float))
        arrays["sr"].append(tab["sigma_r"].astype(float))
        arrays["sz"].append(tab["sigma_z"].astype(float)
                            if "sigma_z" in tab.dtype.names else np.zeros(n))
        arrays["cluster"].append(np.full(n,p["name"]))
        arrays["stage"].append(p["stage"])
    return {k:np.concatenate(v) for k,v in arrays.items()}


def fit_tier(config, tier):
    settings = config["forward_model"]
    parts = load_training(config,tier)
    rng = np.random.default_rng(settings["rng_seed"] +
                                {"gr_all":0,"gr_common":1,"grz_common":2}[tier])
    models = []
    ncolor = 2 if tier == "grz_common" else 1
    for i in range(settings["shared_nuisance_draws"]):
        a = joined(parts,rng,settings["shared_zero_point_sigma_mag"],perturb=i>0)
        color = a["color"][:,:ncolor]
        beta, intrinsic = fit_gls(a["mg"],a["feh"],color,a["sg"],a["sr"],a["sz"],
                                  a["cluster"],ridge=settings["ridge_on_metallicity_slope"],
                                  floor=settings["intrinsic_color_sigma_floor_mag"])
        models.append({"beta":beta.tolist(),"intrinsic":intrinsic.tolist()})
    reference = joined(parts,rng,settings["shared_zero_point_sigma_mag"],perturb=False)
    edge = settings["support_mg_bin_edges"]
    support = support_bins({(p["name"],stage):p["mg"][p["stage"]==stage]
                            for p in parts for stage in p["stage"]},
                           edge,settings["minimum_each_cluster_per_stage_bin"])
    counts = {p["name"]:{stage:int(np.sum(p["stage"]==stage))
                         for stage in np.unique(p["stage"])} for p in parts}
    return {"training_counts":counts,"n":len(reference["mg"]),
            "support":support,"models":models,
            "training_source_ids_sha256":sha_ids(reference["cluster"],parts)}


def sha_ids(clusters, parts):
    # Cluster name and Gaia ID together identify every training source.
    lines = [f"{p['name']}:{int(sid)}" for p in parts for sid in p["tab"]["source_id"]]
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(CONFIG.read_text())
    if "ngc5466" in config["development"]:
        raise ValueError("Blind cluster entered development configuration")
    results = {tier:fit_tier(config,tier)
               for tier in ("gr_all","gr_common","grz_common")}
    result = {"experiment":config["experiment"],
              "test_cluster": "ngc5466_not_read_or_used",
              "configuration_sha256":sha256(CONFIG),
              "input_sha256":{name:sha256(Path(f"data/cluster_rgb_gr_v2/{name}.fits"))
                              for name in config["development"]},
              "tiers":results,"compute_host":socket.gethostname()}
    write_json(OUTPUT,result)
    print(json.dumps({tier:{"n":results[tier]["n"],
                            "support":results[tier]["support"]} for tier in results}),flush=True)


if __name__ == "__main__":
    main()

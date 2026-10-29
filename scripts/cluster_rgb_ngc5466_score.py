#!/usr/bin/env python3
"""Score the previously registered whole-cluster RGB transfer once."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photdis.cluster_rgb_conditional import (grid_from_bins, log_gaussian,
    model_mean, model_slope, posterior_on_grid, quantiles,
    residual_measurement_cov)
from photdis.stage1 import require_compute, write_json

CONFIG = Path("configs/cluster_rgb_conditional_v1.json")
MODEL = Path("manifests/cluster_rgb_conditional_model_v1.json")
BLIND = Path("data/cluster_rgb_gr_v2/ngc5466.fits")
OUTPUT = Path("manifests/cluster_rgb_ngc5466_blind_score_v1.json")


def predictive_log_density(models, mg, feh, weights, observed, sg, sr, sz):
    terms = []
    nc = len(observed)
    for model in models:
        beta = np.asarray(model["beta"])
        intrinsic = np.asarray(model["intrinsic"])
        slope = model_slope(beta,mg)
        cov = residual_measurement_cov(sg,sr,sz,slope,nc)+intrinsic
        for z,w in zip(feh,weights):
            mean = model_mean(beta,np.asarray(mg),z)
            terms.append(float(log_gaussian(np.asarray(observed)-mean,cov))
                         +np.log(w/len(models)))
    peak = max(terms)
    return float(peak+np.log(np.exp(np.asarray(terms)-peak).sum()))


def describe(records):
    if not records:
        return {"n":0}
    residual = np.asarray([r["delta_mu"] for r in records])
    center = np.median(residual)
    out = {"n":len(records),"delta_mu_median":float(center),
           "delta_mu_nmad":float(1.4826*np.median(np.abs(residual-center))),
           "abs_delta_mu_gt_0p5_fraction":float(np.mean(np.abs(residual)>.5)),
           "coverage_68":float(np.mean([r["covered_68"] for r in records])),
           "coverage_95":float(np.mean([r["covered_95"] for r in records])),
           "mean_log_predictive_density":float(np.mean([r["log_predictive_density"]
                                                        for r in records])),
           "posterior_mu_68_width_median":float(np.median([r["mu_68_width"]
                                                             for r in records]))}
    return out


def score_row(row, stage, tier, model, cfg, components, weights):
    cells = model["support"][stage]
    grid = grid_from_bins(cells,cfg["forward_model"]["posterior_grid_step_mag"])
    if not len(grid):
        return None
    truth_mg = float(row["g0"])-cfg["whole_cluster_test"]["ngc5466"]["external_mu"]
    if not any(lo <= truth_mg < hi for lo,hi,*_ in cells):
        return None
    ncolor = 2 if tier == "grz_common" else 1
    observed = [float(row["gr0"])] + ([float(row["rz0"])] if ncolor == 2 else [])
    sg,sr,sz = [float(row[k]) for k in ("sigma_g","sigma_r","sigma_z")]
    probability = posterior_on_grid(model["models"],grid,components,weights,
                                    observed,sg,sr,sz)
    q = quantiles(grid,probability)
    g0 = float(row["g0"])
    mu_pred = g0-q[2]
    mu_lo95,mu_hi95 = g0-q[4],g0-q[0]
    mu_lo68,mu_hi68 = g0-q[3],g0-q[1]
    mu_ref = cfg["whole_cluster_test"]["ngc5466"]["external_mu"]
    return {"source_id":int(row["source_id"]),"stage":stage,
            "mg_ref":truth_mg,"mu_pred":float(mu_pred),
            "delta_mu":float(mu_pred-mu_ref),
            "mu_lo68":float(mu_lo68),"mu_hi68":float(mu_hi68),
            "mu_lo95":float(mu_lo95),"mu_hi95":float(mu_hi95),
            "covered_68":bool(mu_lo68<=mu_ref<=mu_hi68),
            "covered_95":bool(mu_lo95<=mu_ref<=mu_hi95),
            "mu_68_width":float(mu_hi68-mu_lo68),
            "log_predictive_density":predictive_log_density(
                model["models"],truth_mg,components,weights,observed,sg,sr,sz)}


def score_tier(tab, tier, model, cfg):
    stages = tab["phase"].astype("U")
    quality = "gr_clean" if tier == "gr_all" else "grz_clean_same_row"
    finite = np.isfinite(tab["g0"]+tab["gr0"]+tab["sigma_g"]+tab["sigma_r"])
    if tier == "grz_common":
        finite &= np.isfinite(tab["rz0"]+tab["sigma_z"])
    selected = ((tab["pmember"]>.9) & tab[quality] & finite
                & np.isin(stages,("upper_rgb","lower_rgb")))
    test_cfg = cfg["whole_cluster_test"]["ngc5466"]
    sigma = test_cfg["feh_sigma"]
    offset = np.sqrt(2)*sigma
    conditions = {
        "known_rrab":([test_cfg["feh_uves_rrab"]-offset,
                       test_cfg["feh_uves_rrab"],
                       test_cfg["feh_uves_rrab"]+offset],[.25,.5,.25]),
        "known_rrc_sensitivity":([test_cfg["feh_rrc_alternative"]-offset,
                                  test_cfg["feh_rrc_alternative"],
                                  test_cfg["feh_rrc_alternative"]+offset],[.25,.5,.25])}
    two = cfg["unknown_feh_components_uves"]
    for weight in cfg["unknown_feh_prior_weights"]:
        conditions[f"unknown_{weight[0]:.1f}_{weight[1]:.1f}"] = (two,weight)
    by_condition, per_star = {}, {}
    for label,(components,weights) in conditions.items():
        records = []
        for row,stage in zip(tab[selected],stages[selected]):
            score = score_row(row,stage,tier,model,cfg,components,weights)
            if score is not None:
                records.append(score)
        by_stage = {stage:describe([r for r in records if r["stage"]==stage])
                    for stage in ("upper_rgb","lower_rgb")}
        edges = cfg["forward_model"]["support_mg_bin_edges"]
        by_bin = {f"{lo:.1f}_{hi:.1f}":describe([
            r for r in records if lo<=r["mg_ref"]<hi])
            for lo,hi in zip(edges[:-1],edges[1:])}
        by_condition[label] = {"overall":describe(records),
                               "by_stage":by_stage,"by_mg_bin":by_bin}
        per_star[label] = records
    return {"selected_n":int(selected.sum()),
            "supported_n":len(per_star["known_rrab"]),
            "conditions":by_condition},per_star


def plot(path, rows):
    fig,ax=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    for tier,color in (("gr_common","tab:blue"),("grz_common","tab:orange")):
        records=rows[tier]["known_rrab"]
        if records:
            ax[0].scatter([r["mg_ref"] for r in records],
                          [r["delta_mu"] for r in records],s=12,alpha=.55,
                          label=f"{tier} n={len(records)}",color=color)
            ax[1].hist([r["delta_mu"] for r in records],bins=np.arange(-2,2.05,.1),
                       histtype="step",label=tier,color=color)
    ax[0].axhline(0,color="0.4",lw=1)
    ax[0].set(xlabel="Reference M_g",ylabel="Predicted − RR Lyrae modulus (mag)")
    ax[1].set(xlabel="Predicted − RR Lyrae modulus (mag)",ylabel="Stars")
    for a in ax:
        a.grid(alpha=.2)
        a.legend(fontsize=8)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cfg=json.loads(CONFIG.read_text())
    frozen=json.loads(MODEL.read_text())
    if frozen["configuration_sha256"] != hashlib.sha256(CONFIG.read_bytes()).hexdigest():
        raise ValueError("Frozen model configuration mismatch")
    tab=fitsio.read(str(BLIND))
    if set(np.unique(tab["release"])) != {11010}:
        raise ValueError("NGC 5466 test has an unexpected camera release")
    training_ids=set()
    for name in cfg["development"]:
        training_ids.update(map(int,fitsio.read(f"data/cluster_rgb_gr_v2/{name}.fits",
                                                 columns=["source_id"])["source_id"]))
    if training_ids & set(map(int,tab["source_id"])):
        raise ValueError("Blind and development source identities overlap")
    summary,rows={},{}
    for tier,model in frozen["tiers"].items():
        summary[tier],rows[tier]=score_tier(tab,tier,model,cfg)
    if summary["gr_common"]["supported_n"] != summary["grz_common"]["supported_n"]:
        raise ValueError("Same-star band comparison changed supported count")
    p=Path("data/cluster_rgb_conditional_v1/ngc5466_scores.fits")
    p.parent.mkdir(parents=True,exist_ok=True)
    detailed=[]
    for tier in rows:
        for condition,records in rows[tier].items():
            for r in records:
                detailed.append((tier,condition,r["source_id"],r["mg_ref"],
                                 r["mu_pred"],r["delta_mu"],r["mu_lo95"],r["mu_hi95"]))
    if detailed:
        fitstable=np.array(detailed,dtype=[("tier","S16"),("condition","S32"),
            ("source_id","i8"),("mg_ref","f8"),("mu_pred","f8"),
            ("delta_mu","f8"),("mu_lo95","f8"),("mu_hi95","f8")])
        fitsio.write(str(p),fitstable)
    figure=Path("figures/cluster_rgb/06_ngc5466_blind_transfer_v1.png")
    plot(figure,rows)
    result={"role":"registered_new_whole_cluster_test",
            "model_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            "reference_mu":cfg["whole_cluster_test"]["ngc5466"]["external_mu"],
            "shared_reference_mu_sigma":cfg["whole_cluster_test"]["ngc5466"]["external_mu_sigma"],
            "same_star_gr_grz":True,"tier_results":summary,
            "gr_only_added_clean_n":int(((tab["pmember"]>.9)&tab["gr_clean"]
                                         &~tab["grz_clean_same_row"]
                                         &np.isin(tab["phase"].astype("U"),
                                                    ("upper_rgb","lower_rgb"))).sum()),
            "per_star_file_nersc":str(p),"figure":str(figure),
            "compute_host":socket.gethostname()}
    write_json(OUTPUT,result)
    print(json.dumps({"common":summary["grz_common"]["conditions"]["known_rrab"]["overall"],
                      "gr_only_added":result["gr_only_added_clean_n"]}),flush=True)


if __name__=="__main__":
    main()

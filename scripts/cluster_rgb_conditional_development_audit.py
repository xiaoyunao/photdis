#!/usr/bin/env python3
"""Score frozen conditional color curves on their own development systems."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/".deps"),str(ROOT/"scripts")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cluster_rgb_conditional_fit import load_training
from photdis.cluster_rgb_conditional import model_mean
from photdis.stage1 import require_compute, write_json


def describe(values):
    a=np.asarray(values,float)
    if len(a)==0:
        return {"n":0}
    m=np.median(a)
    return {"n":len(a),"median":float(m),
            "nmad":float(1.4826*np.median(abs(a-m))),
            "p05":float(np.percentile(a,5)),"p95":float(np.percentile(a,95))}


def main():
    require_compute()
    out=Path("manifests/cluster_rgb_conditional_development_audit_v1.json")
    if out.exists():
        raise FileExistsError(out)
    cfg=json.loads(Path("configs/cluster_rgb_conditional_v1.json").read_text())
    frozen=json.loads(Path("manifests/cluster_rgb_conditional_model_v1.json").read_text())
    result={}
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    colors={"m3":"tab:blue","m53":"tab:orange","ngc7006":"tab:green"}
    for tier in ("gr_all","gr_common","grz_common"):
        parts=load_training(cfg,tier)
        model=frozen["tiers"][tier]["models"][0]
        beta=np.asarray(model["beta"])
        stats={}
        for p in parts:
            mg=p["mg"]
            tab=p["tab"]
            obs=np.column_stack((tab["gr0"],tab["rz0"]))[:,:len(beta)]
            pred=model_mean(beta,mg,p["feh"])
            residual=obs-pred
            stats[p["name"]]={}
            for stage in np.unique(p["stage"]):
                mask=p["stage"]==stage
                stats[p["name"]][stage]={
                    "mg":describe(mg[mask]),
                    "gr":describe(residual[mask,0]),
                    **({"rz":describe(residual[mask,1])} if len(beta)==2 else {})}
                if tier=="grz_common":
                    for j in range(2):
                        axes[j].scatter(mg[mask],residual[mask,j],s=5,alpha=.3,
                                        color=colors[p["name"]],
                                        label=f"{p['name']} {stage}" if j==0 else None)
        result[tier]=stats
    for axis,label in zip(axes,("gr", "rz")):
        axis.axhline(0,color="0.3",lw=1)
        axis.set(xlabel="Reference M_g",ylabel=f"Observed − model {label} (mag)")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7,markerscale=2)
    figure=Path("figures/cluster_rgb/06_conditional_development_residuals_v1.png")
    figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure,dpi=160)
    plt.close(fig)
    write_json(out,{"frozen_model":"manifests/cluster_rgb_conditional_model_v1.json",
                    "development_residuals":result,"figure":str(figure),
                    "compute_host":socket.gethostname()})
    print(json.dumps({k:{name:{stage:v["gr"]["median"] for stage,v in stages.items()}
                          for name,stages in stats.items()}
                      for k,stats in result.items()}),flush=True)


if __name__=="__main__":
    main()

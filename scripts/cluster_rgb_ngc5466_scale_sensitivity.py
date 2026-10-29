#!/usr/bin/env python3
"""Coherent test-cluster dust and per-band zero-point perturbations."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/".deps"),str(ROOT/"scripts")]
import fitsio
import numpy as np

from cluster_rgb_ngc5466_score import score_row
from photdis.cluster_rgb_conditional import RG,RR,RZ
from photdis.stage1 import require_compute,write_json


def score(tab, model, cfg, feh, weight):
    result={}
    for row in tab:
        stage=row["phase"].decode().strip()
        if not (row["pmember"]>.9 and row["grz_clean_same_row"]
                and stage in ("upper_rgb","lower_rgb")):
            continue
        r=score_row(row,stage,"grz_common",model,cfg,feh,weight)
        if r is not None:
            result[r["source_id"]]=r
    return result


def perturb(tab, kind, delta):
    a=tab.copy()
    if kind=="dust":
        a["g0"]-=RG*delta
        a["gr0"]-=(RG-RR)*delta
        a["rz0"]-=(RR-RZ)*delta
    elif kind=="g_zero":
        a["g0"]+=delta
        a["gr0"]+=delta
    elif kind=="r_zero":
        a["gr0"]-=delta
        a["rz0"]+=delta
    elif kind=="z_zero":
        a["rz0"]-=delta
    else:
        raise ValueError(kind)
    return a


def main():
    require_compute()
    out=Path("manifests/cluster_rgb_ngc5466_scale_sensitivity_v1.json")
    if out.exists():
        raise FileExistsError(out)
    cfg=json.loads(Path("configs/cluster_rgb_conditional_v1.json").read_text())
    model=json.loads(Path("manifests/cluster_rgb_conditional_model_v1.json").read_text())["tiers"]["grz_common"]
    tab=fitsio.read("data/cluster_rgb_gr_v2/ngc5466.fits")
    tcfg=cfg["whole_cluster_test"]["ngc5466"]
    offset=np.sqrt(2)*tcfg["feh_sigma"]
    feh=[tcfg["feh_uves_rrab"]-offset,tcfg["feh_uves_rrab"],
         tcfg["feh_uves_rrab"]+offset]
    weights=[.25,.5,.25]
    baseline=score(tab,model,cfg,feh,weights)
    cases={}
    for kind,amplitude in (("dust",tcfg["dust_scale_sigma_ebv"]),
                           ("g_zero",cfg["forward_model"]["shared_zero_point_sigma_mag"]),
                           ("r_zero",cfg["forward_model"]["shared_zero_point_sigma_mag"]),
                           ("z_zero",cfg["forward_model"]["shared_zero_point_sigma_mag"])):
        for sign in (-1,1):
            delta=sign*amplitude
            changed=score(perturb(tab,kind,delta),model,cfg,feh,weights)
            common=sorted(set(baseline)&set(changed))
            difference=np.asarray([changed[sid]["delta_mu"]-baseline[sid]["delta_mu"]
                                   for sid in common])
            cases[f"{kind}_{sign:+d}"]={
                "delta":delta,"n_common":len(common),
                "n_support_changed":len(set(baseline)^set(changed)),
                "median_delta_mu_change":float(np.median(difference)) if len(common) else None,
                "p05_p95_delta_mu_change":list(map(float,np.percentile(difference,[5,95])))
                    if len(common) else []}
    result={"baseline_n":len(baseline),"cases":cases,
            "external_mu_shared_sigma":tcfg["external_mu_sigma"],
            "note":"all perturbations are coherent across the test cluster; no model refit",
            "compute_host":socket.gethostname()}
    write_json(out,result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()

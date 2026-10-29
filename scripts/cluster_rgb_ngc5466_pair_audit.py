#!/usr/bin/env python3
"""Paired same-source uncertainty for the frozen NGC 5466 gr/grz comparison."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/".deps")]
import fitsio
import numpy as np

from photdis.stage1 import require_compute, write_json


def nmad(x):
    m=np.median(x)
    return 1.4826*np.median(abs(x-m))


def paired(tab):
    tier=tab["tier"].astype("U")
    condition=tab["condition"].astype("U")
    a=tab[(tier=="gr_common")&(condition=="known_rrab")]
    b=tab[(tier=="grz_common")&(condition=="known_rrab")]
    a=a[np.argsort(a["source_id"])]
    b=b[np.argsort(b["source_id"])]
    if len(a)!=len(b) or not np.array_equal(a["source_id"],b["source_id"]):
        raise ValueError("Same-source comparison identity mismatch")
    return a,b


def main():
    require_compute()
    out=Path("manifests/cluster_rgb_ngc5466_pair_audit_v1.json")
    if out.exists():
        raise FileExistsError(out)
    tab=fitsio.read("data/cluster_rgb_conditional_v1/ngc5466_scores.fits")
    a,b=paired(tab)
    ra,rb=a["delta_mu"],b["delta_mu"]
    if not len(a):
        raise ValueError("No common supported blind-test sources")
    rng=np.random.default_rng(5466001)
    draws=[]
    for _ in range(4000):
        idx=rng.integers(0,len(a),len(a))
        draws.append(nmad(ra[idx])-nmad(rb[idx]))
    low=rb[(b["mg_ref"]>=1.0)&(b["mg_ref"]<1.5)]
    high=rb[(b["mg_ref"]>=2.5)&(b["mg_ref"]<3.0)]
    result={"n_common":len(a),
            "nmad_gr":float(nmad(ra)),"nmad_grz":float(nmad(rb)),
            "nmad_gr_minus_grz":float(nmad(ra)-nmad(rb)),
            "source_bootstrap_95_nmad_gain":list(map(float,np.percentile(draws,[2.5,97.5]))),
            "median_gr":float(np.median(ra)),"median_grz":float(np.median(rb)),
            "median_paired_gr_minus_grz":float(np.median(ra-rb)),
            "low_mg_1_to_1p5_n":len(low),"low_mg_median":float(np.median(low)) if len(low) else None,
            "high_mg_2p5_to_3_n":len(high),"high_mg_median":float(np.median(high)) if len(high) else None,
            "high_minus_low_median":float(np.median(high)-np.median(low)) if len(low) and len(high) else None,
            "interpretation":"source bootstrap measures within this single test cluster; the external distance modulus is one shared scale parameter",
            "compute_host":socket.gethostname()}
    write_json(out,result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()

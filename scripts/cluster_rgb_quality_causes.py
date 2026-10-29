#!/usr/bin/env python3
"""Attribute existing clean-grz losses by DR11 mask and fraction fields."""
from __future__ import annotations

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

SYSTEMS = ("m3", "m53", "ngc2419", "ngc7006")
MASK_BITS = {"satur_g": 2, "satur_r": 3, "satur_z": 4,
             "allmask_g": 5, "allmask_r": 6, "allmask_z": 7,
             "bailout": 10}


def causes(tab, members):
    base = members & tab["grz_base"]
    clean = base & tab["grz_clean"]
    frac_fail = base & ~tab["grz_fraction_clean"]
    mask_fail = base & ~tab["grz_mask_clean"]
    out = {"members": int(members.sum()), "grz_base": int(base.sum()),
           "grz_clean": int(clean.sum()), "frac_fail": int(frac_fail.sum()),
           "mask_fail": int(mask_fail.sum()),
           "both_frac_mask_fail": int((frac_fail & mask_fail).sum()),
           "maskbit_causes_among_base": {name: int((base &
               ((tab["maskbits"].astype(np.int64) & (1 << bit)) != 0)).sum())
               for name, bit in MASK_BITS.items()}}
    bands = {}
    for band in ("g", "r", "z"):
        fm = tab[f"fracmasked_{band}"]
        fi = tab[f"fracin_{band}"]
        ff = tab[f"fracflux_{band}"]
        bands[band] = {
            "fracmasked_ge_0p4_or_invalid": int((base & ~(np.isfinite(fm) & (fm < .4))).sum()),
            "fracin_le_0p7_or_invalid": int((base & ~(np.isfinite(fi) & (fi > .7))).sum()),
            "fracflux_ge_0p5_or_invalid": int((base & ~(np.isfinite(ff) & (ff < .5))).sum())}
    out["fraction_causes_among_base"] = bands
    return out


def main():
    require_compute()
    output = Path("manifests/cluster_rgb_quality_causes_v1.json")
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(CONFIG.read_text())["systems"]
    result = {}
    for name in SYSTEMS:
        tab = fitsio.read(str(Path("data") / f"cluster_rgb_{name}_v1" / "members.fits"))
        inv = json.loads((Path("manifests") / f"cluster_rgb_{name}_inventory_v1.json").read_text())
        phase = classify(tab["gaia_g"]-inv["external_modulus"]["mu_mag"],
                         tab["bp_rp"], config[name])
        result[name] = {kind: causes(tab, (tab["pmember"] > .9) & (phase == kind))
                        for kind in ("upper_rgb", "lower_rgb", "hb", "possible_agb")}
    write_json(output, {"systems": result, "maskbits": MASK_BITS,
                        "compute_host": socket.gethostname(),
                        "note": "Cause counts overlap; reserved system has quality counts only"})
    print(json.dumps({name: {phase: stats["grz_clean"] for phase, stats in phases.items()}
                      for name, phases in result.items()}), flush=True)


if __name__ == "__main__":
    main()

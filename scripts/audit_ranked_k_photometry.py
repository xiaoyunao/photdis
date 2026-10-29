#!/usr/bin/env python3
"""Locate optical-quality loss among ranked spectroscopic K giants."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import numpy as np

from photdis.stage1 import require_compute, write_json

SOURCE = Path("data/giant_ranked_vac_v1/same_star.fits")
CONFIG = Path("configs/joint_giant_ranked_fields.json")
OUTPUT = Path("manifests/giant_ranked_k_photometry_v1.json")


def median(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def main():
    require_compute()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = fitsio.read(str(SOURCE))
    config = json.loads(CONFIG.read_text())
    names = np.char.strip(rows["field"].astype("U"))
    result = {}
    for entry in config["fields"]:
        field = json.loads(Path(entry["config"]).read_text())
        k = rows[(names == field["name"])
                 & (rows["desi_teff"] >= 4200) & (rows["desi_teff"] < 5000)]
        phot = fitsio.read(str(Path(field["output_dir"]) / "field_diagnostics.fits"),
                           columns=["ls_id", "grz_base", "grz_fraction_clean",
                                    "grz_mask_clean", "grz_clean", "g0"])
        order = np.argsort(phot["ls_id"])
        sorted_id = phot["ls_id"][order]
        position = np.searchsorted(sorted_id, k["ls_id"])
        if np.any(position >= len(sorted_id)) or not np.array_equal(
                sorted_id[np.minimum(position, len(sorted_id)-1)], k["ls_id"]):
            raise ValueError(f"Legacy identity mismatch in {field['name']}")
        matched = phot[order[position]]
        base = matched["grz_base"]
        fraction = matched["grz_fraction_clean"]
        mask = matched["grz_mask_clean"]
        clean = matched["grz_clean"]
        if not np.array_equal(clean, k["grz_clean"]):
            raise AssertionError("Joint and field clean flags disagree")
        result[field["name"]] = {
            "role": entry["role"], "k_giants": int(len(k)),
            "grz_base": int(base.sum()),
            "fraction_clean": int(fraction.sum()),
            "mask_clean": int(mask.sum()),
            "grz_clean": int(clean.sum()),
            "fraction_fail_mask_pass": int((~fraction & mask).sum()),
            "mask_fail_fraction_pass": int((fraction & ~mask).sum()),
            "both_fraction_and_mask_fail": int((~fraction & ~mask).sum()),
            "g0_median_clean": median(matched["g0"][clean]),
            "g0_median_rejected": median(matched["g0"][~clean]),
        }
    write_json(OUTPUT, {"source": str(SOURCE), "by_field": result,
                        "compute_host": socket.gethostname()})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot the preregistered six-field frozen native-residual census."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/cold_v4_native_transfer_sky_census.json"
MANIFEST = ROOT / "manifests/cold_v4_native_transfer_sky_census.json"
OUTPUT = ROOT / "figures/cold_v4/09_native_sky_census.png"


def main():
    registered = json.loads(REGISTRATION.read_text())
    scored = json.loads(MANIFEST.read_text())
    if scored["registration_sha256"] != hashlib.sha256(
            REGISTRATION.read_bytes()).hexdigest():
        raise ValueError("Sky-census result does not match registration")
    fields = {field["field"]: field for field in scored["fields"]}
    if set(fields) != {item["name"] for item in registered["diagnostic_fields"]}:
        raise ValueError("Sky-census field set differs from registration")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for item in registered["diagnostic_fields"]:
        field = fields[item["name"]]
        b = item["center_galactic_latitude_deg"]
        color = "#2459a6" if b >= 0 else "#b66128"
        cases = [
            (field["distance_modulus_residual"], field["sky_block_bootstrap"]),
            (field["by_group"]["parallax_ge_2_mas"],
             field["parallax_group_sky_block_bootstrap"]["parallax_ge_2_mas"]),
        ]
        for axis, (value, ci) in zip(axes, cases):
            if value["n"] < 50 or ci is None:
                continue
            median = value["median"]
            low, high = ci["median_95pct"]
            axis.errorbar(b, median, yerr=np.array([[median - low],
                          [high - median]]), fmt="o", color=color,
                          capsize=3, ms=6)
            axis.annotate(f"RA {item['bounds_deg'][0]}\nn={value['n']:,}",
                          (b, median), xytext=(4, 5),
                          textcoords="offset points", fontsize=8)
    for axis, title in zip(axes, ("All selected cold anchors",
                                   r"Corrected parallax $\geq 2$ mas")):
        axis.axhline(0, color="black", lw=.7)
        axis.axvline(0, color="gray", lw=.7, alpha=.4)
        axis.set_title(title)
        axis.set_xlabel("Center Galactic latitude (deg)")
        axis.set_ylabel(r"Median $\hat\mu-\mu_{\rm Gaia}$ (mag)")
        axis.grid(alpha=.15)
    fig.suptitle("Six preregistered diagnostic fields; frozen cold-v4 route")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()

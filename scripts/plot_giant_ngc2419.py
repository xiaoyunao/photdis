#!/usr/bin/env python3
"""Plot NGC 2419 member distance VAC checks from the small manifest."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/giant_ngc2419_reference_v1.json"
OUT = ROOT / "figures/giant_vac/02_ngc2419_catalog_scale.png"


def main():
    report = json.loads(SOURCE.read_text())
    stars = [star for star in report["stars"] if star["grz_clean"]]
    mu = report["reference"]["mu_mag"]
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.5))
    for ax, name, label, color in (
        (axes[0], "spdist", "SPDist", "#226b91"),
        (axes[1], "specdis", "SpecDis", "#a04b29")):
        selected = [star for star in stars if f"{name}_mu" in star]
        teff = np.array([star["teff"] for star in selected])
        delta = np.array([star[f"{name}_mu"] - mu for star in selected])
        error = np.array([star[f"{name}_sigma_mu"] for star in selected])
        ax.errorbar(teff, delta, yerr=error, fmt="o", color=color,
                    ecolor=color, capsize=2, alpha=.82, ms=5)
        ax.axhline(0, color="#242424", lw=1)
        ax.axhspan(-.08, .08, color="#777777", alpha=.13)
        ax.axvline(5000, ls="--", color="#777777", lw=1)
        ax.set_xlabel("DESI RVS temperature (K)")
        ax.set_ylabel(r"Catalog $\mu$ − NGC 2419 RR Lyrae $\mu$ (mag)")
        ax.set_title(f"{label}: {len(selected)} clean members")
        ax.grid(alpha=.18)
    fig.suptitle("NGC 2419: whole-system RR Lyrae reference, μ=19.71±0.08 mag")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot the registered warm-giant reserved-field and M3 scale checks."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path("manifests/giant_photo_baseline_v1.json")
OUTPUT = Path("figures/giant_vac/03_warm_photo_baseline.png")


def main():
    data = json.loads(SOURCE.read_text())
    strata = data["reserved_by_physical_stratum"]
    names = ["Teff 5000–5500", "Teff 5500–7000", "log g < 2.5",
             "log g ≥ 2.5", "[Fe/H] < −1.5", "[Fe/H] ≥ −1.5"]
    keys = ["teff_5000_5500", "teff_5500_7000", "logg_lt2p5",
            "logg_ge2p5", "feh_lt_minus1p5", "feh_ge_minus1p5"]
    center = [strata[k]["median_mg_residual"] for k in keys]
    scatter = [strata[k]["nmad_mg_residual"] for k in keys]
    counts = [strata[k]["n"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    y = np.arange(len(keys))
    axes[0].barh(y, center, color="#4c78a8")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_yticks(y, [f"{name} (n={n})" for name, n in zip(names, counts)])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Median predicted − SPDist $M_g$ (mag)")
    for i, value in enumerate(scatter):
        axes[0].text(0.31, i, f"NMAD {value:.3f}", va="center", fontsize=8)
    axes[0].set_xlim(-0.35, 0.75)
    axes[0].set_title("Reserved fields: different evolved populations")

    reference = data["m3_external"]["reference_mu_mag"]
    members = data["m3_external"]["members"]
    for i, member in enumerate(members):
        lo, hi = member["mu_68"]
        median = member["mu_median"]
        axes[1].errorbar(median, i, xerr=[[median - lo], [hi - median]],
                         fmt="o", color="#e45756", capsize=2)
    axes[1].axvline(reference, color="black", lw=1.2,
                    label=f"RR Lyrae $\\mu$={reference:.3f}")
    axes[1].axvline(data["m3_external"]["median_photo_mu_mag"],
                    color="#e45756", ls="--", lw=1.1,
                    label="Median photo-only prediction")
    axes[1].set_yticks(range(len(members)), [str(i + 1) for i in range(len(members))])
    axes[1].set_xlabel("M3 distance modulus (mag)")
    axes[1].set_ylabel("Qualified member")
    axes[1].set_title("Whole-system holdout: M3")
    axes[1].legend(fontsize=8, loc="lower right")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

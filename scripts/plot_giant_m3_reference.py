#!/usr/bin/env python3
"""Plot M3 member distance labels against its RR Lyrae system distance."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path("manifests/giant_m3_reference_v1.json")
OUT = Path("figures/giant_vac/01_m3_external_distance.png")


def main():
    data = json.loads(SOURCE.read_text())
    members = data["members"]
    reference = data["reference"]
    mu = reference["mu_mag"]
    sigma = np.hypot(reference["statistical_sigma_mag"],
                     reference["systematic_sigma_mag"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    ax.axhspan(-sigma, sigma, color="#dddddd", alpha=.8,
               label="M3 RR Lyrae system distance uncertainty")
    ax.axhline(0, color="black", lw=.8)
    for name, shift, color in (("specdis", -.06, "#3274a1"),
                               ("spdist", .06, "#bd6c36")):
        values = [(i, row[f"{name}_mu"]-mu, row[f"{name}_mu_error"])
                  for i, row in enumerate(members) if f"{name}_mu" in row]
        if not values:
            continue
        x, y, error = np.asarray(values).T
        ax.errorbar(x+shift, y, yerr=error, fmt="o", color=color,
                    capsize=2.5, label=name.upper())
    ax.set_xticks(range(len(members)), [str(i+1) for i in range(len(members))])
    ax.set_xlabel("Qualified DESI giant member")
    ax.set_ylabel("Catalog distance modulus minus M3 RR Lyrae value (mag)")
    ax.set_title("M3 (NGC 5272): whole-system external distance check")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8, loc="best")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot held-out parallax-density gains from the bounded iron term."""
from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path("manifests/metal_poor_forward_v1.json")
OUT = Path("figures/metal_poor_forward/01_reserved_parallax_logdensity.png")


def main():
    data = json.loads(SOURCE.read_text())
    plain = data["models"]["no_feh"]
    iron = data["models"]["one_feh"]
    fields = list(iron["reserved_by_field"])
    names = ["RA" + re.search(r"ra(\d{3})_", name).group(1) for name in fields]
    labels = ["All reserved", "Blue reserved", "Poor reserved", "Blue + poor reserved"]
    groups = ["all", "blue_gr0_lt_0p909", "poor_feh_lt_m1p5", "blue_poor"]
    delta = [iron["reserved_by_field"][name]["loglike_mean"]
             - plain["reserved_by_field"][name]["loglike_mean"] for name in fields]
    sizes = [iron["reserved_by_field"][name]["n"] for name in fields]
    group_delta = [iron["reserved"][name]["loglike_mean"]
                   - plain["reserved"][name]["loglike_mean"] for name in groups]
    group_sizes = [iron["reserved"][name]["n"] for name in groups]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), layout="constrained")
    for ax, names_here, values, counts, title, color in (
        (axes[0], names, delta, sizes, "By held-out sky field", "#3577ad"),
        (axes[1], labels, group_delta, group_sizes, "By held-out subset", "#bc6b31")):
        ypos = np.arange(len(names_here))
        ax.barh(ypos, values, color=color, alpha=.84)
        ax.set_yticks(ypos, names_here)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=.8)
        ax.set_xlabel("Mean log predictive density gain per star")
        ax.set_title(title)
        ax.set_xlim(left=min(-.03, min(values)-.03), right=max(values)*1.23)
        ax.grid(axis="x", alpha=.2)
        for i, (value, n) in enumerate(zip(values, counts)):
            ax.text(value + .015, i, f"n={n:,}", va="center", fontsize=8)
    fig.suptitle("One continuous [Fe/H] term versus color-only forward parallax model")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot diagnostic blue-side colors and Gaia luminosity from NERSC inventory."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SOURCE = Path("data/cold_v5_blue/inventory.json")
OUTPUT = Path("figures/cold_v5/01_blue_inventory.png")


def points(fields, label):
    return [row for field in fields.values() for row in
            field["plot_samples"][label]]


def coordinates(rows, x, y):
    return (np.array([row[x] for row in rows if row[x] is not None and
                      row[y] is not None], float),
            np.array([row[y] for row in rows if row[x] is not None and
                      row[y] is not None], float))


def main():
    inventory = json.loads(SOURCE.read_text())
    fields = inventory["fields"]
    blue = points(fields, "gaia_blue")
    dwarf = points(fields, "cool_dwarf")
    poor = points(fields, "calibrated_FeH_lt_minus_1p5_dwarf")
    giant = points(fields, "cool_giant")
    nonstar = points(fields, "known_QSO_or_galaxy")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for rows, name, color, size, alpha in (
            (blue, "Gaia anchor candidate", "0.7", 5, .3),
            (dwarf, "DESI cool dwarf", "tab:blue", 9, .5),
            (poor, "DESI [Fe/H]<−1.5 dwarf", "tab:purple", 25, .85),
            (giant, "DESI cool giant", "tab:orange", 16, .6),
            (nonstar, "DESI known nonstar", "tab:red", 9, .4)):
        x, y = coordinates(rows, "gr0", "rz0")
        axes[0].scatter(x, y, s=size, c=color, alpha=alpha, label=name,
                        rasterized=True)
    axes[0].axvline(inventory["frozen_gr_lower_edge"], color="black", lw=1,
                    ls="--", label="v4 g−r lower edge")
    axes[0].axhline(.401, color="black", lw=1, ls=":")
    axes[0].set(xlabel="(g−r)₀", ylabel="(r−z)₀",
                title="Blue of the frozen cold-v4 box")
    axes[0].legend(fontsize=7, loc="best")
    for rows, name, color, size, alpha in (
            (blue, "Gaia anchor candidate", "0.5", 6, .3),
            (poor, "DESI poor dwarf with Gaia Mg", "tab:purple", 28, .9)):
        x, y = coordinates(rows, "gr0", "mg")
        axes[1].scatter(x, y, s=size, c=color, alpha=alpha, label=name,
                        rasterized=True)
    axes[1].axvline(inventory["frozen_gr_lower_edge"], color="black", lw=1,
                    ls="--")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="(g−r)₀", ylabel="Gaia-derived Legacy M_g",
                title="Available high-S/N parallax anchors")
    axes[1].legend(fontsize=7)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

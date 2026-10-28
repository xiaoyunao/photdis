#!/usr/bin/env python3
"""Plot DESI giant and metal-poor dwarf positions against the frozen optical box."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/cold_v4_type_boundary.json"
OUTPUT = ROOT / "figures/cold_v4/05_type_boundary.png"


def numeric(rows, key):
    return np.array([row[key] if row[key] is not None else np.nan for row in rows], float)


def main():
    data = json.loads(SOURCE.read_text())
    fields = list(data["fields"].values())
    dwarf = [item for row in fields for item in row["background_sample"]["cold_dwarf"]]
    giant = [item for row in fields for item in row["accepted_cold_giant_objects"]]
    poor = [item for row in fields for item in row["optical_box_rejected_poor_dwarf_objects"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis in axes:
        axis.grid(alpha=.18)
    axes[0].scatter(numeric(dwarf, "gr0"), numeric(dwarf, "rz0"),
                    s=5, alpha=.16, color="#56616a", label="DESI cool dwarfs (sample)")
    axes[0].scatter(numeric(giant, "gr0"), numeric(giant, "rz0"), s=40,
                    marker="D", facecolor="#cf5b36", edgecolor="black", linewidth=.4,
                    label=f"Accepted cool giants (n={len(giant)})")
    axes[0].scatter(numeric(poor, "gr0"), numeric(poor, "rz0"), s=60,
                    marker="x", linewidth=1.7, color="#225ca8",
                    label=f"Box-rejected [Fe/H]<−1.5 dwarfs (n={len(poor)})")
    low, high = data["optical_box_low"], data["optical_box_high"]
    axes[0].add_patch(Rectangle((low[0], low[1]), high[0]-low[0],
                                high[1]-low[1], fill=False, linestyle="--",
                                linewidth=1.3, edgecolor="black"))
    axes[0].set_xlabel("Dereddened g−r")
    axes[0].set_ylabel("Dereddened r−z")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    for rows, label, color, marker, size in (
            (dwarf, "DESI cool dwarfs", "#56616a", ".", 6),
            (giant, "Accepted cool giants", "#cf5b36", "D", 38),
            (poor, "Box-rejected poor dwarfs", "#225ca8", "x", 56)):
        x, y = numeric(rows, "rz0"), numeric(rows, "legacy_mg_from_gaia_anchor"
                                      if rows is not dwarf else "mg")
        good = np.isfinite(x) & np.isfinite(y)
        axes[1].scatter(x[good], y[good], s=size, marker=marker, color=color,
                        alpha=.18 if rows is dwarf else .9, label=f"{label} (n={good.sum()})")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Dereddened r−z")
    axes[1].set_ylabel("Legacy M_g from high-quality Gaia (mag)")
    axes[1].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

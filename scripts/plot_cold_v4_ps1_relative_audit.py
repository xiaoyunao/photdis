#!/usr/bin/env python3
"""Plot the matched-PS1 optical and frozen-route sensitivity checks."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/cold_v4_ps1_relative_audit_final.json"
OUTPUT = ROOT / "figures/cold_v4/11_ps1_relative_audit.png"
NAMES = ("South RA330", "South RA20", "North RA130")
COLORS = ("#bd612b", "#db9b32", "#2459a6")


def point(axis, x, stats, blocks, color, label=None, marker="o"):
    middle = stats["median"]
    low, high = blocks["median_95pct"]
    axis.errorbar(x, middle, yerr=np.array([[middle - low], [high - middle]]),
                  fmt=marker, ms=5, capsize=3, color=color, label=label)


def main():
    result = json.loads(MANIFEST.read_text())
    fields = result["fields"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for index, (field, color) in enumerate(zip(fields, COLORS, strict=True)):
        for band_index, band in enumerate("grz"):
            optical = field["optical"][band]
            point(axes[0], band_index + (index - 1) * .18,
                  optical["legacy_minus_transformed_ps1_mag"],
                  optical["sky_block"], color, NAMES[index] if band == "g" else None)
        for axis, key in zip(axes[1:],
                             ("replacement_all", "replacement_parallax_ge_2_mas")):
            row = field[key]
            point(axis, index - .09, row["legacy_residual"],
                  row["legacy_residual_sky_block"], color,
                  "Legacy" if index == 0 else None)
            point(axis, index + .09, row["ps1_replacement_residual"],
                  row["ps1_replacement_residual_sky_block"], color,
                  "PS1 optical substitution" if index == 0 else None,
                  marker="s")
    for axis in axes:
        axis.axhline(0, color="black", lw=.7)
        axis.grid(alpha=.2)
    axes[0].set_xticks(range(3), list("grz"))
    axes[0].set_title("Legacy − transformed PS1")
    axes[0].set_ylabel("Observed-band difference (mag)")
    axes[0].legend(fontsize=8)
    for axis, title in zip(axes[1:],
                           ("Same PS1-qualified stars", r"Same stars, $\varpi_c\geq2$ mas")):
        axis.set_xticks(range(3), NAMES, rotation=20)
        axis.set_title(title)
        axis.set_ylabel(r"Median $\hat\mu-\mu_{\rm Gaia}$ (mag)")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("PS1 relative optical comparison; frozen Legacy-trained forest")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()

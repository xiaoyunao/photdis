#!/usr/bin/env python3
"""Plot the frozen cold-v4 distance response to scaled foreground extinction."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/cold_v4_dust_sensitivity_final.json"
OUTPUT = ROOT / "figures/cold_v4/10_dust_sensitivity.png"
LABELS = (
    ("South RA 330–340", "#bd612b"),
    ("South RA 20–30", "#db9b32"),
    ("North RA 130–140", "#2459a6"),
)


def main():
    result = json.loads(MANIFEST.read_text())
    fractions = result["assumed_foreground_fractions"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for field, (label, color) in zip(result["fields"], LABELS, strict=True):
        records = [field["fractions"][f"fraction_{fraction:g}"]
                   for fraction in fractions]
        all_median = [entry["same_source_assumed_residual"]["median"]
                      for entry in records]
        near_median = [entry["parallax_ge_2_mas"]
                       ["same_source_assumed_residual"]["median"]
                       for entry in records]
        axes[0].plot(fractions, all_median, "o-", color=color, label=label)
        axes[1].plot(fractions, near_median, "o-", color=color, label=label)
    for ax, title in zip(axes, ("All accepted cold anchors",
                                r"Corrected parallax $\geq 2$ mas")):
        ax.axhline(0, color="black", lw=.7)
        ax.set_xlim(-.03, 1.03)
        ax.set_xticks(fractions)
        ax.set_xlabel("Fraction of catalog foreground extinction subtracted")
        ax.set_ylabel(r"Median $\hat\mu-\mu_{\rm Gaia}$ (mag)")
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen cold-v4 route: bounded dust sensitivity on the same fields")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()

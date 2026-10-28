#!/usr/bin/env python3
"""Plot application tier/quality mix and strict-Gaia interval coverage."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/cold_v4_application_audit.json"
OUTPUT = ROOT / "figures/cold_v4/02_application_tiers.png"
TIERS = ("A_grz", "B_grzW1", "C_grzW1W2")
QUALITY = ("clean_precise_optical", "clean_loose_optical", "base_nonclean")


def main():
    manifest = json.loads(MANIFEST.read_text())
    fields = [value for key, value in manifest["fields"].items()
              if key.startswith("cold_v4_final_")]
    if len(fields) != 3:
        raise ValueError("Expected three registered cold-v4 final fields")
    mix = np.array([[sum(field["photometric_candidate_distribution"]["tier_x_quality"][tier][quality]
                         for field in fields) for quality in QUALITY] for tier in TIERS])
    cover = np.array([[field["cold_Gaia_metrics"][f"{tier}_strict_anchor"]
                        ["final_distance_modulus"]["coverage_95"] for tier in TIERS]
                      for field in fields])
    if int(mix.sum()) != 443280:
        raise ValueError("Published application count changed")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    colors = ("#3d6b83", "#ddaa44", "#be5d58")
    labels = ("clean, optical σ<0.03", "clean, less precise", "base, nonclean")
    bottom = np.zeros(3)
    for index in range(3):
        values = mix[:, index] / 1000
        axes[0].bar(["A grz", "B grzW1", "C grzW1W2"], values, bottom=bottom,
                    color=colors[index], label=labels[index])
        bottom += values
    axes[0].set_ylabel("Routed photometric candidates (thousands)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=.2)
    for index, row in enumerate(cover):
        axes[1].plot(range(3), row * 100, marker="o", lw=1.8,
                     label=fields[index]["field"].split("_ra")[1].split("_dec")[0])
    axes[1].axhline(95, color=".4", ls="--", lw=1, label="nominal 95%")
    axes[1].set_xticks(range(3), ["A grz", "B grzW1", "C grzW1W2"])
    axes[1].set_ylim(80, 100)
    axes[1].set_ylabel("95% modulus coverage on strict Gaia cold dwarfs (%)")
    axes[1].legend(title="RA field", frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

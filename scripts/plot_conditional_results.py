#!/usr/bin/env python3
"""Plot frozen final-field metrics with field counts and coverage uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TIERS = ("A_grz", "B_grzW1", "C_grzW1W2")
COLORS = ("#356e9b", "#58894e", "#bf6f38")


def wilson95(success_fraction, n):
    z = 1.96
    center = (success_fraction + z*z/(2*n)) / (1 + z*z/n)
    radius = z * np.sqrt(success_fraction*(1-success_fraction)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return center-radius, center+radius


def plot(summary, path):
    fields = list(summary["fields"])
    x = np.arange(len(fields), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for index, (tier, color) in enumerate(zip(TIERS, COLORS)):
        x_tier = x + (index - 1) * .10
        rows = [summary["fields"][field]["tiers"][tier] for field in fields]
        axes[0].scatter(x_tier, [row["center_Mg"]["bias"] for row in rows],
                        color=color, s=48, label=tier)
        coverage = np.array([row["center_and_group_cqr_Mg"]["coverage_95"]
                             for row in rows])
        n = np.array([row["center_and_group_cqr_Mg"]["n"] for row in rows])
        ci = np.array([wilson95(value, count) for value, count in zip(coverage, n)])
        axes[1].errorbar(x_tier, coverage,
                         yerr=[coverage-ci[:, 0], ci[:, 1]-coverage],
                         fmt="o", color=color, capsize=3)
        axes[1].scatter(x_tier, [row["direct_Mg"]["coverage_95"] for row in rows],
                        facecolors="none", edgecolors=color, marker="s", s=45, alpha=.65)
    axes[0].axhline(0, color=".5", ls=":")
    axes[1].axhline(.95, color=".5", ls=":")
    labels = []
    for field in fields:
        data = summary["fields"][field]["field"]
        r0, _, r1, _ = data["bounds_deg"]
        labels.append(f"RA {r0:03d}–{r1:03d}\nN={data['cold_strict_anchor_rows']:,}")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlim(-.35, len(fields)-.65)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Centered predicted − Gaia $M_g$ [mag]")
    axes[1].set_ylabel("95% $M_g$ interval coverage")
    axes[1].set_ylim(.75, 1.005)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Filled: calibrated ± Wilson 95%; open: direct")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplemental", action="store_true")
    args = parser.parse_args()
    if args.supplemental:
        manifest = Path("manifests/conditional_supplemental_v2.json")
        figure = Path("figures/conditional_v2/02_supplemental_conditional_calibration.png")
    else:
        manifest = Path("manifests/conditional_final_v2.json")
        figure = Path("figures/conditional_v2/01_final_conditional_calibration.png")
    plot(json.loads(manifest.read_text()), figure)


if __name__ == "__main__":
    main()

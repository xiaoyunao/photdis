#!/usr/bin/env python3
"""Show template-field and noise-field effects in the paired deep experiment."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/cold_v4_crossover.json"
OUTPUT = ROOT / "figures/cold_v4/06_template_noise_crossover.png"


def main():
    result = json.loads(SOURCE.read_text())
    x = np.arange(4)
    labels = ["18–19", "19–20", "20–21", "21–22"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    colors = ("#155c87", "#b4552d")
    linestyles = ("-", "--")
    for template in (0, 1):
        for environment in (0, 1):
            rows = [result["cells"][f"{template}|{environment}|{depth}"]
                    for depth in range(4)]
            label = f"template {template} × noise {environment}"
            style = {"color": colors[template], "linestyle": linestyles[environment],
                     "marker": "o", "label": label}
            axes[0].plot(x, [row["components"]["degraded_total"]["median"]
                             for row in rows], **style)
            axes[1].plot(x, [row["components"]["degraded_total"]["nmad"]
                             for row in rows], **style)
            axes[2].plot(x, [row["acceptance"] * 100 for row in rows], **style)
    axes[0].set_ylabel("Median final-modulus residual (mag)")
    axes[1].set_ylabel("Final-modulus NMAD (mag)")
    axes[2].set_ylabel("Full-route acceptance (%)")
    axes[0].axhline(0, color="black", linewidth=.8)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Matched target r bin (mag)")
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Same Gaia templates across matched DR11 noise environments", fontsize=11)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot the preregistered cold-v4 final-field summary from its small manifest."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/cold_v4_final.json"
OUTPUT = ROOT / "figures/cold_v4/02_new_final_fields.png"


def main() -> None:
    fields = json.loads(SOURCE.read_text())["fields"]
    names = list(fields)
    labels = ["RA 0–10", "RA 150–160", "RA 220–230"]
    colors = ["#35649a", "#c47432", "#55844e"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for i, name in enumerate(names):
        field = fields[name]
        metric = field["all_selected"]["final_distance_modulus"]
        ci = field["all_selected"]["sky_block_ci"]["coverage_95_95pct"]
        cover = metric["coverage_95"]
        axes[0].errorbar(i, cover, yerr=[[cover - ci[0]], [ci[1] - cover]],
                         fmt="o", color=colors[i], capsize=4,
                         label=f"{labels[i]}  (n={metric['n']:,})")
        groups = field["application_visible_groups"]
        axes[1].plot(np.arange(6),
                     [group["metrics"]["final_distance_modulus"]["bias"]
                      for group in groups], "o-", color=colors[i], label=labels[i])
    axes[0].axhline(.95, color=".4", ls=":", lw=1)
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylabel("Final distance-modulus 95% coverage")
    axes[0].set_ylim(.90, .99)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].axhline(0, color=".4", ls=":", lw=1)
    axes[1].set_xticks(range(6))
    axes[1].set_xlabel("Application-visible color × brightness group")
    axes[1].set_ylabel("Median distance-modulus residual [mag]")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.15)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

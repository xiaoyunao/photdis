#!/usr/bin/env python3
"""Show shared-noise development routing tradeoff by target depth."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/cold_v4_policy_development.json"
OUTPUT = ROOT / "figures/cold_v4/03_policy_depth.png"
POLICIES = (("clean", "highest supported + clean"),
            ("clean_wise_snr_20", "clean + WISE S/N ≥20"),
            ("strict", "strict tier quality"))


def main():
    data = json.loads(SOURCE.read_text())
    depths = data["depths"]
    if len(depths) != 4:
        raise ValueError("Expected four fixed target depth bins")
    x = [sum(row["r_bin"]) / 2 for row in depths]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for policy, label in POLICIES:
        rows = [row["policies"][policy] for row in depths]
        axes[0].plot(x, [row["acceptance"] * 100 for row in rows], "o-", label=label)
        axes[1].plot(x, [row["direct"]["distance_modulus_mag"]["nmad"]
                         for row in rows], "o-", label=label)
    axes[0].set_ylabel("Accepted shared-noise observations (%)")
    axes[1].set_ylabel("Final distance-modulus NMAD (mag)")
    for axis in axes:
        axis.set_xlabel("Target observed r (mag)")
        axis.set_xticks(x, [f"{row['r_bin'][0]}–{row['r_bin'][1]}" for row in depths])
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

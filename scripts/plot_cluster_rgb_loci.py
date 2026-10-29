#!/usr/bin/env python3
"""Plot inspectable development-cluster Legacy RGB median sequences."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {"m3": ("M3 lower RGB", "tab:blue"),
           "m53": ("M53 bridge", "tab:orange"),
           "ngc2419": ("NGC 2419 upper RGB", "tab:green")}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5), constrained_layout=True)
    for name, (label, color) in SYSTEMS.items():
        manifest = json.loads((ROOT / "manifests" /
                               f"cluster_rgb_{name}_phase_v1.json").read_text())
        for ax, key, xlabel in zip(axes, ("gr0", "rz0"),
                                   ("(g−r)₀", "(r−z)₀")):
            points = manifest["locus_by_legacy_mg"][key]
            if name == "ngc2419":
                points = [p for p in points if p["mg_bin"][0] < .5]
            for point in points:
                ax.plot([point["color_p16"], point["color_p84"]],
                        [point["mg_median"]]*2, color=color, alpha=.55, lw=2)
            ax.plot([p["color_median"] for p in points],
                    [p["mg_median"] for p in points], "o-", color=color,
                    ms=4, lw=1.4, label=label)
            ax.set(xlabel=xlabel, ylabel="Legacy M_g = g₀ − cluster μ")
            ax.grid(alpha=.2)
    for ax in axes:
        ax.set_ylim(3.1, -2.7)
        ax.legend(fontsize=8)
    fig.suptitle("Measured clean Legacy RGB candidates; horizontal lines are within-bin color 16–84%")
    out = ROOT / "figures/cluster_rgb/03_development_loci.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

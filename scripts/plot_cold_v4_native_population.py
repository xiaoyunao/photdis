#!/usr/bin/env python3
"""Plot descriptive native residual strata for the two crossover fields."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/cold_v4_native_population.json"
OUTPUT = ROOT / "figures/cold_v4/07_native_population.png"


def main():
    fields = json.loads(SOURCE.read_text())["fields"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    definitions = (("p_corr", ["1–2", "≥2"], [2, 3], "Corrected Gaia parallax (mas)"),
                   ("gaia_bp_rp", ["<1.5", "1.5–2.3", "2.3–3.0", "≥3.0"],
                    [0, 1, 2, 3], "Gaia BP−RP (mag)"))
    for axis, (name, labels, indices, xlabel) in zip(axes, definitions):
        x = np.arange(len(indices))
        for field, offset, color in (("0", -.12, "#155c87"),
                                     ("1", .12, "#b4552d")):
            bins = fields[field]["residual_by_property"][name]
            use = [bins[i]["residual"] for i in indices]
            medians = [row.get("median", np.nan) for row in use]
            axis.scatter(x + offset, medians, marker="o", s=58, color=color,
                         label="RA35–45" if field == "0" else "RA170–180")
            for xx, row in zip(x + offset, use):
                axis.annotate(str(row["n"]), (xx, row.get("median", np.nan)),
                              xytext=(0, 7), textcoords="offset points",
                              ha="center", fontsize=7, color=color)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_xticks(x, labels)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Median native distance-modulus residual (mag)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Same frozen cold-v4 route; numbers show source counts", fontsize=11)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

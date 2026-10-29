#!/usr/bin/env python3
"""Plot the preregistered same-star development comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path("manifests/cold_v5_xd_comparison.json")
OUTPUT = Path("figures/cold_v5/02_xd_same_star_comparison.png")


def main():
    report = json.loads(SOURCE.read_text())
    fields = [(name, field) for name, field in report["fields"].items()
              if name in ("dev_cal_ra205_215_dec_m010_p000",
                          "native_dev_ra060_070_dec_m035_m025")]
    if not fields:
        raise ValueError("No development fields in comparison")
    names = ["RA" + name.split("ra")[-1].split("_")[0] for name, _ in fields]
    x = np.arange(len(fields))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    specs = (("center_mag", "Median residual (mag)"),
             ("nmad_mag", "Residual NMAD (mag)"),
             ("coverage95", "Raw 95% interval coverage"))
    for ax, (metric, label) in zip(axes.flat[:3], specs):
        for model, color, offset in (("qrf", "tab:blue", -.09),
                                     ("xd", "tab:orange", .09)):
            values = [field[model][metric] for _, field in fields]
            ax.scatter(x + offset, values, label=model.upper(), color=color, s=45)
            ax.plot(x + offset, values, color=color, alpha=.45)
        ax.set(xticks=x, xticklabels=names, ylabel=label)
        ax.legend(fontsize=8)
    axes[0, 0].axhline(0, color="0.5", lw=.8)
    axes[1, 0].axhline(.95, color="0.5", lw=.8, ls="--")
    ax = axes[1, 1]
    for j, band in enumerate(("g", "r")):
        for model, color, marker, offset in (("qrf", "tab:blue", "o", -.09),
                                             ("xd", "tab:orange", "s", .09)):
            values = [field["perturbation_plus_0p01_mag"][band][model]
                      ["median_abs_mag"] for _, field in fields]
            ax.scatter(x + offset + j*.4, values, label=f"{model.upper()} {band}",
                       color=color, marker=marker, s=42)
    ax.set(xticks=np.column_stack([x, x+.4]).ravel(),
           xticklabels=[f"{name} {band}" for name in names
                        for band in ("g", "r")],
           ylabel="Median absolute Δμ for +0.01 mag (mag)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"Cold-dwarf v4 vs XD K={report['components']}: identical development stars")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

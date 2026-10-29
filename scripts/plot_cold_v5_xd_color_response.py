#!/usr/bin/env python3
"""Plot same-star optical response by observed dereddened color."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path("manifests/cold_v5_xd_color_response.json")
OUTPUT = Path("figures/cold_v5/03_xd_color_response.png")
FIELDS = (("dev_cal_ra205_215_dec_m010_p000", "RA205–215"),
          ("native_dev_ra060_070_dec_m035_m025", "RA60–70"))


def main():
    report = json.loads(SOURCE.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True,
                             sharey=True, constrained_layout=True)
    for row_index, (field_name, title) in enumerate(FIELDS):
        groups = report["fields"][field_name]["by_color"]
        occupied = [(name, value) for name, value in groups.items()
                    if value["n"] > 0]
        x = np.array([float(name.split("_")[0]) / 2 +
                      float(name.split("_")[1]) / 2 for name, _ in occupied])
        for col_index, band in enumerate(("g", "r")):
            ax = axes[row_index, col_index]
            for model, color in (("qrf", "tab:blue"), ("xd", "tab:orange")):
                y = [value["perturbation_plus_0p01_mag"][band][model]
                     ["median_abs_mag"] for _, value in occupied]
                ax.plot(x, y, marker="o", label=model.upper(), color=color)
            ax.set(title=f"{title}: +0.01 mag {band}",
                   ylabel="Median absolute Δμ (mag)" if col_index == 0 else None,
                   xlabel="Observed (g−r)₀" if row_index == 1 else None)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle("Frozen v4 and XD K=32: identical stars, fixed-tier optical response")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

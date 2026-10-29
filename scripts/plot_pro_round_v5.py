#!/usr/bin/env python3
"""Selected plots of the latest Pro mechanism checks."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures/pro_round_v5")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cross = json.loads(Path("manifests/cold_v4_crossover_common.json").read_text())
    weak = json.loads(Path("manifests/cold_v4_weak_parallax_v3.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for t, color in ((0, "tab:blue"), (1, "tab:orange")):
        rows = [cross["cells"][f"{t}|{d}"]["common"] for d in range(4)]
        x = np.arange(4) + (-.05 if t == 0 else .05)
        mid = np.array([r["difference_of_medians"] for r in rows])
        interval = np.array([r["difference_of_medians_bootstrap95"] for r in rows])
        axes[0].errorbar(x, mid, yerr=np.array([mid-interval[:, 0], interval[:, 1]-mid]),
                         fmt="o", color=color, label=f"Template {t}: difference of medians")
        axes[0].plot(x, [r["median_of_paired_differences"] for r in rows],
                     "x--", color=color, alpha=.7)
    axes[0].axhline(0, color="grey", lw=.7)
    axes[0].set(xticks=range(4), xticklabels=["18–19", "19–20", "20–21", "21–22"],
                xlabel="Target r bin", ylabel="Environment 1 − 0 (mag)",
                title="Centers and paired shifts differ")
    axes[0].legend(fontsize=7)
    rows = [f["wise20_route"]["all"]["selected_center_null"] for f in weak["fields"].values()
            if "all" in f["wise20_route"]]
    for i, row in enumerate(rows):
        lo, hi = row["null_95_mas"]
        axes[1].plot([lo, hi], [i, i], color="tab:blue", lw=5, alpha=.4)
        axes[1].scatter(row["null_median_mas"], i, color="tab:blue", marker="|", s=150)
        axes[1].scatter(row["observed_mas"], i, color="black", marker="x", s=65)
    axes[1].set(yticks=range(3), yticklabels=["RA205 development", "RA150", "RA220"],
                xlabel="Median(model mean − observed parallax), mas",
                title="Observed centers within selected-model null")
    fig.savefig(OUT / "01_statistics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

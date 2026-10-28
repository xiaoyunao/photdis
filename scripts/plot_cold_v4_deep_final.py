#!/usr/bin/env python3
"""Plot final-field modulus interval diagnostics by measured r bin."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/cold_v4_deep_final_v2.json"
OUTPUT = ROOT / "figures/cold_v4/04_deep_final_observed_r.png"


def main():
    data = json.loads(SOURCE.read_text())
    rows = [data["observed_r_bins"][str(i)] for i in range(4)]
    x = np.arange(4)
    labels = [f"{lo}–{hi}" for row in rows for lo, hi in [row["observed_r_bin"]]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, key, nominal in zip(axes[:2], ("coverage_68", "coverage_95"), (.68, .95)):
        direct = np.array([row["tiers"]["all"]["direct_mu"][key] for row in rows])
        calibrated = np.array([row["tiers"]["all"]["calibrated_mu"][key] for row in rows])
        ci_key = "coverage_68_95pct" if nominal == .68 else "coverage_95_95pct"
        ci = np.array([row["tiers"]["all"]["calibrated_source_cluster_ci"][ci_key]
                       for row in rows])
        axis.plot(x, direct * 100, "o--", color="#7d8ca0", label="direct")
        axis.errorbar(x, calibrated * 100,
                      yerr=np.stack((np.maximum(0, calibrated - ci[:, 0]) * 100,
                                     np.maximum(0, ci[:, 1] - calibrated) * 100)),
                      fmt="o-", capsize=3, color="#0d6986", label="calibrated")
        axis.axhline(nominal * 100, color="#a84d2b", linewidth=1, alpha=.8)
        axis.set_ylabel(f"{int(nominal * 100)}% interval coverage (%)")
        axis.set_ylim(max(0, nominal * 100 - 25), 100)
        axis.grid(alpha=.2)
    direct = [row["tiers"]["all"]["direct_mu"]["median_width_95"] for row in rows]
    calibrated = [row["tiers"]["all"]["calibrated_mu"]["median_width_95"] for row in rows]
    axes[2].plot(x, direct, "o--", color="#7d8ca0", label="direct")
    axes[2].plot(x, calibrated, "o-", color="#0d6986", label="calibrated")
    axes[2].set_ylabel("Median 95% width (mag)")
    axes[2].grid(alpha=.2)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Final measured r (mag)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Independent final field · 4 draws per source · measured r bins", fontsize=11)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

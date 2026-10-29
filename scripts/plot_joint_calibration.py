#!/usr/bin/env python3
"""Plot the bounded joint-sample color, abundance and luminosity overlap."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".deps")]
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photdis.stage1 import require_compute

SOURCE = Path("data/joint_calibration_v1/same_star.fits")
OUTPUT = Path("figures/joint_calibration/01_color_metallicity_luminosity.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require_compute()
    a = fitsio.read(str(args.source), columns=["class_evidence", "clean_low_dust",
        "feh_cal_valid", "feh_cal", "gr0", "mg_high_snr", "parallax_layer"])
    labels = np.char.strip(a["class_evidence"].astype("U"))
    layers = np.char.strip(a["parallax_layer"].astype("U"))
    core = ((labels == "cool_main_sequence") &
            a["clean_low_dust"] & a["feh_cal_valid"] &
            np.isfinite(a["gr0"]) & np.isfinite(a["feh_cal"]))
    high = core & (layers == "high_ge20")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    axes[0].scatter(a["gr0"][core], a["feh_cal"][core], s=4, alpha=.2,
                    color="#5a8198", rasterized=True,
                    label=f"clean core with [Fe/H]: {core.sum():,}")
    axes[0].scatter(a["gr0"][high], a["feh_cal"][high], s=18,
                    facecolors="none", edgecolors="#c35832", linewidths=.8,
                    label=f"Gaia S/N≥20: {high.sum():,}")
    axes[0].axvline(.909, color="#555", linestyle="--", linewidth=1)
    axes[0].axhline(-1.5, color="#777", linestyle=":", linewidth=1)
    axes[0].set(xlabel=r"$(g-r)_0$", ylabel=r"DESI calibrated [Fe/H]",
                xlim=(.35, 1.65), ylim=(-3.5, .7),
                title="Spectral labels and geometric anchors")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    p = axes[1].scatter(a["gr0"][high], a["mg_high_snr"][high],
                         c=a["feh_cal"][high], s=18, cmap="viridis",
                         vmin=-2, vmax=.4, alpha=.8)
    axes[1].axvline(.909, color="#555", linestyle="--", linewidth=1)
    axes[1].set(xlabel=r"$(g-r)_0$", ylabel=r"$M_g$ from Gaia S/N≥20",
                xlim=(.35, 1.65), ylim=(10, 2),
                title="High-S/N same-star luminosity subset")
    fig.colorbar(p, ax=axes[1], label="DESI calibrated [Fe/H]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()

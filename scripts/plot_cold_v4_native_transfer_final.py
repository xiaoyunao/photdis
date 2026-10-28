#!/usr/bin/env python3
"""Plot the frozen native-transfer development and preregistered final scores."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("manifests/cold_v4_native_transfer_development.json", 0, "S dev RA60"),
    ("manifests/cold_v4_native_transfer_supplemental2.json", 0, "N dev RA230"),
    ("manifests/cold_v4_native_transfer_final.json", 1, None),
    ("manifests/cold_v4_native_transfer_north_final.json", 1, "N final RA130"),
]


def load_fields():
    fields = []
    for source, final, label in SOURCES:
        for field in json.loads((ROOT / source).read_text())["fields"]:
            if field["selected_count"] < 100:
                continue
            name = label or ("S final RA330" if field["bounds_deg"][0] == 330
                             else "S final RA20")
            fields.append((name, final, field))
    return fields


def main():
    fields = load_fields()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    color = {0: "#6b7280", 1: "#2459a6"}
    for axis, key, title in [
        (axes[0], None, "All selected cold anchors"),
        (axes[1], "parallax_ge_2_mas", r"Corrected parallax $\geq 2$ mas"),
    ]:
        for index, (name, final, field) in enumerate(fields):
            if key is None:
                value = field["distance_modulus_residual"]
                ci = field["sky_block_bootstrap"]
            else:
                value = field["by_group"][key]
                ci = field.get("parallax_group_sky_block_bootstrap", {}).get(key)
            center = value["median"]
            if ci is None:
                axis.plot(center, index, "o", color=color[final], ms=6)
            else:
                low, high = ci["median_95pct"]
                axis.errorbar(center, index, xerr=np.array([[center - low],
                    [high - center]]), fmt="o", color=color[final], ms=6,
                    capsize=3, lw=1.4)
            axis.text(.155, index, f"n={value['n']:,}", va="center", fontsize=8)
        axis.axvline(0, color="black", lw=.7, alpha=.6)
        axis.set_xlim(-.105, .205)
        axis.set_title(title)
        axis.set_xlabel(r"Median $\hat\mu-\mu_{\rm Gaia}$ (mag)")
        axis.grid(axis="x", alpha=.15)
    axes[0].set_yticks(range(len(fields)), [item[0] for item in fields])
    axes[0].invert_yaxis()
    fig.suptitle("Frozen cold-v4 native transfer across registered sky fields")
    fig.tight_layout()
    output = ROOT / "figures/cold_v4/08_native_transfer_final.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()

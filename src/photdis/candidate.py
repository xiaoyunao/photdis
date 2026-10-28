"""Application-visible selection and risk flags for cold-dwarf hypotheses."""

from __future__ import annotations

import numpy as np

from .support import supported


def flags(table, rule, tree):
    """Return color-box, support and input-tier flags without Gaia or spectra."""
    colors = np.column_stack((table["gr0"], table["rz0"])).astype(float)
    finite = np.all(np.isfinite(colors), axis=1)
    low = np.asarray(rule["color_1pct"], float)
    high = np.asarray(rule["color_99pct"], float)
    box = np.asarray(table["grz_base"], bool) & finite & np.all(
        (colors >= low) & (colors <= high), axis=1)
    support = np.zeros(len(colors), bool)
    support[box] = supported(tree, colors[box], rule["support_scale_mag"],
                             rule["support_10th_neighbor_95pct"], neighbors=10)
    tier = np.zeros(len(colors), np.uint8)
    tier[box] = 1
    tier[box & np.asarray(table["grz_w1"], bool)] = 2
    tier[box & np.asarray(table["grz_w1_w2"], bool)] = 3
    return {"color_box": box, "support_ok": support, "tier": tier,
            "strict_optical": np.asarray(table["grz_high_quality"], bool),
            "r_ge_20": np.asarray(table["r"], float) >= 20,
            "type_unverified": np.ones(len(colors), bool)}

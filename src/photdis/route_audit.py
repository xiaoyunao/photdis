"""Explain missing higher photometric tiers without altering the frozen route."""
from __future__ import annotations

import numpy as np

from .route import LEVEL


def _band_reason(table, band, threshold):
    size = len(table["gr0"])
    reason = np.full(size, "usable", dtype="U24")
    valid = np.asarray(table[f"valid_{band}"], bool)
    snr = np.asarray(table[f"snr_{band}"], float)
    clean = np.asarray(table[f"{band}_clean"], bool)
    reason[~valid] = "missing_or_invalid"
    reason[valid & (~np.isfinite(snr) | (snr < threshold))] = "low_snr"
    reason[valid & (snr >= threshold) & ~clean] = "quality_failed"
    return reason


def higher_tier_reasons(table, routed, chosen, threshold=20):
    """Report why B/C are unavailable; a supported lower tier remains unchanged."""
    size = len(chosen)
    if len(table["gr0"]) != size:
        raise ValueError("Route and photometry lengths differ")
    w1 = _band_reason(table, "w1", threshold)
    w2 = _band_reason(table, "w2", threshold)
    b = w1.copy()
    c = np.where(w1 == "usable", w2, w1)
    b[(b == "usable") & ~routed["support_by_tier"]["B_grzW1"]] = "support_conflict"
    c[(c == "usable") & ~routed["support_by_tier"]["C_grzW1W2"]] = "support_conflict"
    active = np.asarray(chosen) > 0
    conflict = (active & (((chosen == LEVEL["A_grz"]) &
                         (b == "support_conflict")) |
                         ((chosen == LEVEL["B_grzW1"]) &
                          (c == "support_conflict"))))
    return {"B": b, "C": c, "higher_support_conflict": conflict}

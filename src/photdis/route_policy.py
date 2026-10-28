"""Development-only quality and WISE threshold policies for a frozen route."""

from __future__ import annotations

import numpy as np

from .route import LEVEL


def choose_policy(table, routed, policy):
    """Choose the highest eligible supported tier without changing its support.

    A policy is one of ``available``, ``clean``, ``strict``, or
    ``clean_wise_snr_{5,10,20}``. The WISE threshold applies to every WISE
    band used by the chosen tier. Optical clean flags are mandatory in the
    latter three threshold policies.
    """
    allowed = {"available", "clean", "strict", "clean_wise_snr_5",
               "clean_wise_snr_10", "clean_wise_snr_20"}
    if policy not in allowed:
        raise ValueError(f"Unknown route policy: {policy}")
    size = len(routed["tier"])
    level = np.zeros(size, np.uint8)
    clean = np.asarray(table["grz_clean"], bool)
    threshold = int(policy.rsplit("_", 1)[-1]) if policy.startswith("clean_wise_snr_") else None
    strict_flags = {"A_grz": "grz_high_quality",
                    "B_grzW1": "grz_w1_high_quality",
                    "C_grzW1W2": "full_high_quality"}
    for tier in reversed(tuple(LEVEL)):
        eligible = np.asarray(routed["support_by_tier"][tier], bool).copy()
        if policy == "strict":
            eligible &= np.asarray(table[strict_flags[tier]], bool)
        elif policy != "available":
            eligible &= clean
        if threshold is not None and tier != "A_grz":
            eligible &= np.asarray(table["snr_w1"], float) >= threshold
            if tier == "C_grzW1W2":
                eligible &= np.asarray(table["snr_w2"], float) >= threshold
        level[(level == 0) & eligible] = LEVEL[tier]
    return level

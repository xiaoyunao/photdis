"""Feature-matched training support and photometric tier routing."""

from __future__ import annotations

import numpy as np

from .followup import apply_cqr, cqr_offset
from .support import support_radius, supported


FEATURES = {
    "A_grz": ("gr0", "rz0"),
    "B_grzW1": ("gr0", "rz0", "zw10"),
    "C_grzW1W2": ("gr0", "rz0", "zw10", "w1w20"),
}
LAYER = {"A_grz": "grz_base", "B_grzW1": "grz_w1", "C_grzW1W2": "grz_w1_w2"}
LEVEL = {"A_grz": 1, "B_grzW1": 2, "C_grzW1W2": 3}


def feature_matrix(table, tier):
    """Read only application-visible dereddened colors."""
    return np.column_stack([np.asarray(table[name], float) for name in FEATURES[tier]])


def fit_route(training, scale_by_tier, neighbors=10, quantile=.95):
    """Fit optical box and tier-specific kNN support on model fit rows."""
    if set(training) != set(FEATURES):
        raise ValueError("Training colors must include all three tiers")
    if set(scale_by_tier) != set(FEATURES):
        raise ValueError("Missing tier support scale")
    optical = np.asarray(training["A_grz"], float)
    if optical.ndim != 2 or optical.shape[1] != 2 or not np.isfinite(optical).all():
        raise ValueError("A training colors must be finite optical pairs")
    route = {"optical_low": np.quantile(optical, .01, axis=0),
             "optical_high": np.quantile(optical, .99, axis=0),
             "neighbors": neighbors, "tiers": {}}
    for tier, names in FEATURES.items():
        colors = np.asarray(training[tier], float)
        if colors.ndim != 2 or colors.shape[1] != len(names):
            raise ValueError(f"Wrong training feature dimension for {tier}")
        scale = np.asarray(scale_by_tier[tier], float)
        tree, radius = support_radius(colors, scale, neighbors, quantile)
        route["tiers"][tier] = {"tree": tree, "scale": scale,
                                "radius": radius, "fit_rows": len(colors)}
    return route


def route_rows(table, route):
    """Select optical box then highest available supported tier, C→B→A."""
    optical = feature_matrix(table, "A_grz")
    finite_optical = np.isfinite(optical).all(axis=1)
    box = (np.asarray(table[LAYER["A_grz"]], bool) & finite_optical &
           np.all((optical >= route["optical_low"]) &
                  (optical <= route["optical_high"]), axis=1))
    level = np.zeros(len(optical), np.uint8)
    support = {}
    for tier in reversed(tuple(FEATURES)):
        colors = feature_matrix(table, tier)
        available = box & np.asarray(table[LAYER[tier]], bool) & np.isfinite(colors).all(axis=1)
        accepted = np.zeros(len(optical), bool)
        model = route["tiers"][tier]
        accepted[available] = supported(model["tree"], colors[available],
                                        model["scale"], model["radius"],
                                        neighbors=route["neighbors"])
        support[tier] = accepted
        level[(level == 0) & accepted] = LEVEL[tier]
    return {"color_box": box, "tier": level, "support_by_tier": support,
            "selected": level > 0}


def fit_route_offsets(truth, prediction, groups, min_group=80):
    """Calibrate the complete route; sparse groups use global offsets."""
    truth = np.asarray(truth, float)
    prediction = np.asarray(prediction, float)
    groups = np.asarray(groups, int)
    if truth.ndim != 1 or prediction.shape != (len(truth), 5) or groups.shape != truth.shape:
        raise ValueError("Truth, five quantiles and group labels must align")
    if not np.all((groups >= 0) & (groups < 6)):
        raise ValueError("Route groups must be 0 through 5")
    global68, _ = cqr_offset(truth, prediction, 1, 3, .32)
    global95, _ = cqr_offset(truth, prediction, 0, 4, .05)
    offsets = {}
    for group in range(6):
        take = groups == group
        if take.sum() >= min_group:
            q68, _ = cqr_offset(truth[take], prediction[take], 1, 3, .32)
            q95, _ = cqr_offset(truth[take], prediction[take], 0, 4, .05)
            source = "group"
        else:
            q68, q95 = global68, global95
            source = "global"
        offsets[str(group)] = {"n": int(take.sum()), "q68": q68,
                               "q95": q95, "source": source}
    return offsets


def apply_route_offsets(prediction, groups, offsets):
    """Expand 68% and 95% intervals for the fixed routed prediction."""
    prediction = np.asarray(prediction, float)
    groups = np.asarray(groups, int)
    if prediction.ndim != 2 or prediction.shape[1] != 5 or len(groups) != len(prediction):
        raise ValueError("Prediction and groups must align")
    result = prediction.copy()
    for group in range(6):
        take = groups == group
        if take.any():
            value = offsets[str(group)]
            result[take] = apply_cqr(result[take], value["q68"], value["q95"])
    return result

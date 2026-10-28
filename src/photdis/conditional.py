"""Color/depth center correction and groupwise split-CQR calibration."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from photdis.followup import apply_cqr, cqr_offset


FEATURES = {
    "A_grz": ("gr0", "rz0", "r", "sigma_g", "sigma_r", "sigma_z"),
    "B_grzW1": ("gr0", "rz0", "zw10", "r", "sigma_g", "sigma_r", "sigma_z", "sigma_w1"),
    "C_grzW1W2": ("gr0", "rz0", "zw10", "w1w20", "r", "sigma_g", "sigma_r", "sigma_z", "sigma_w1", "sigma_w2"),
}


def matrix(table, tier):
    """Application-visible photometry, including log errors, in fixed order."""
    names = FEATURES[tier]
    values = []
    for name in names:
        column = np.asarray(table[name], float)
        if name.startswith("sigma_"):
            with np.errstate(divide="ignore", invalid="ignore"):
                column = np.log10(column)
        values.append(column)
    return np.column_stack(values)


def fit_center(x, residual, ridge_alpha=100.0):
    """Fit a regularized correction to predicted-minus-reference Mg residual."""
    x = np.asarray(x, float)
    residual = np.asarray(residual, float)
    good = np.all(np.isfinite(x), axis=1) & np.isfinite(residual)
    if good.sum() < 100:
        raise ValueError("Too few development anchors for center correction")
    scaler = StandardScaler().fit(x[good])
    model = Ridge(alpha=ridge_alpha).fit(scaler.transform(x[good]),
                                         np.clip(residual[good], -1, 1))
    return scaler, model


def correct_center(prediction, x, scaler, model):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError("Center features must be finite")
    correction = model.predict(scaler.transform(x))
    return np.asarray(prediction, float) - correction[:, None]


def group_labels(rz, r, color_edges, r_edge):
    """Six fixed groups: three observed r-z colors by two observed r depths."""
    rz = np.asarray(rz, float)
    r = np.asarray(r, float)
    color = np.searchsorted(np.asarray(color_edges)[1:-1], rz, side="right")
    return color * 2 + (r >= r_edge).astype(int)


def fit_group_cqr(truth, prediction, groups, min_group=80):
    """Finite-sample CQR in fixed groups, with unlabeled-count fallback."""
    truth = np.asarray(truth, float)
    prediction = np.asarray(prediction, float)
    groups = np.asarray(groups, int)
    if len(truth) != len(prediction) or len(truth) != len(groups):
        raise ValueError("Calibration arrays must align")
    global68, _ = cqr_offset(truth, prediction, 1, 3, .32)
    global95, _ = cqr_offset(truth, prediction, 0, 4, .05)
    offsets = {}
    for group in range(6):
        take = groups == group
        n = int(take.sum())
        if n >= min_group:
            q68, _ = cqr_offset(truth[take], prediction[take], 1, 3, .32)
            q95, _ = cqr_offset(truth[take], prediction[take], 0, 4, .05)
            source = "color_x_depth"
        else:
            color_take = groups // 2 == group // 2
            if color_take.sum() >= min_group:
                q68, _ = cqr_offset(truth[color_take], prediction[color_take], 1, 3, .32)
                q95, _ = cqr_offset(truth[color_take], prediction[color_take], 0, 4, .05)
                source = "color_only"
            else:
                q68, q95 = global68, global95
                source = "global"
        offsets[str(group)] = {"n": n, "q68": q68, "q95": q95, "source": source}
    return offsets


def apply_group_cqr(prediction, groups, offsets):
    adjusted = np.asarray(prediction, float).copy()
    groups = np.asarray(groups, int)
    for group in range(6):
        take = groups == group
        if take.any():
            offset = offsets[str(group)]
            adjusted[take] = apply_cqr(adjusted[take], offset["q68"], offset["q95"])
    return adjusted

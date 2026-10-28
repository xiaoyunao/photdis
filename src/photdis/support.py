"""Color-space training support calibrated with consistent neighbor ranks."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def support_radius(training_colors, scale, neighbors=10, quantile=0.95):
    """Return the training percentile of distance to other training objects."""
    training_colors = np.asarray(training_colors, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if len(training_colors) <= neighbors:
        raise ValueError("Support calibration needs more training objects than neighbors")
    if not np.isfinite(training_colors).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Support colors and positive scales must be finite")
    tree = cKDTree(training_colors / scale)
    # The self-query includes the object itself, so request one extra neighbor.
    distances = tree.query(training_colors / scale, k=neighbors + 1)[0][:, -1]
    return tree, float(np.quantile(distances, quantile))


def supported(tree, colors, scale, radius, neighbors=10):
    """Flag targets whose tenth training neighbor is within the radius."""
    colors = np.asarray(colors, dtype=float)
    distance = tree.query(colors / np.asarray(scale, dtype=float), k=neighbors)[0][:, -1]
    return distance <= radius

"""Small, explicit quantile forest for the pilot color--luminosity relation."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

QUANTILES = np.array([.025, .16, .5, .84, .975])


def spatial_partition(cells, eligible, seed=20261028):
    """Assign entire sky cells; balance counts before any model is fitted."""
    cells = np.asarray(cells)
    eligible = np.asarray(eligible, bool)
    unique, counts = np.unique(cells[eligible], return_counts=True)
    if len(unique) < 10:
        raise ValueError("Too few occupied sky cells for three spatial partitions")
    order = np.random.default_rng(seed).permutation(len(unique))
    labels = np.zeros(len(unique), np.int8)
    target = .15 * int(eligible.sum())
    cursor = 0
    for partition in (2, 1):
        assigned = 0
        while assigned < target and cursor < len(order) - 5:
            index = order[cursor]
            labels[index] = partition
            assigned += int(counts[index])
            cursor += 1
    result = np.full(len(cells), 3, np.int8)
    index = np.searchsorted(unique, cells)
    inside = (index < len(unique))
    inside[inside] &= unique[index[inside]] == cells[inside]
    result[inside] = labels[index[inside]]
    if np.any(np.bincount(result[eligible], minlength=4)[:3] < 100):
        raise ValueError("Spatial split has fewer than 100 stars in a model partition")
    return result, {str(int(c)): int(p) for c, p in zip(unique, labels)}


class QuantileForest:
    """Each tree shares weight uniformly among its training leaf members."""

    def __init__(self, n_estimators=300, min_samples_leaf=20, random_state=20261028):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            max_features=.75, bootstrap=False, n_jobs=1, random_state=random_state)

    def fit(self, x, y):
        self.y = np.asarray(y, float)
        self.model.fit(np.asarray(x, float), self.y)
        self.members = []
        for tree in self.model.estimators_:
            leaf = tree.apply(np.asarray(x, float))
            self.members.append({int(value): np.flatnonzero(leaf == value)
                                 for value in np.unique(leaf)})
        return self

    def predict(self, x, quantiles=QUANTILES):
        x = np.asarray(x, float)
        order = np.argsort(self.y)
        results = np.empty((len(x), len(quantiles)), float)
        leaves = self.model.apply(x)
        for i in range(len(x)):
            weight = np.zeros(len(self.y), float)
            for j, member in enumerate(self.members):
                ids = member[int(leaves[i, j])]
                weight[ids] += 1 / (len(self.members) * len(ids))
            positive = weight[order] > 0
            conditional_y = self.y[order][positive]
            cumulative = np.cumsum(weight[order][positive])
            positions = np.searchsorted(cumulative, quantiles, side="left")
            results[i] = conditional_y[np.minimum(positions, len(conditional_y) - 1)]
        return results


def metrics(truth, predicted, g0=None, parallax=None):
    """Residual, robust scatter, interval coverage and optional distance error."""
    truth = np.asarray(truth, float)
    residual = predicted[:, 2] - truth
    middle = float(np.median(residual))
    result = {"n": len(truth), "bias_mag": middle,
              "nmad_mag": float(1.4826 * np.median(np.abs(residual - middle))),
              "large_0p5_fraction": float(np.mean(np.abs(residual) > .5)),
              "large_1p0_fraction": float(np.mean(np.abs(residual) > 1)),
              "coverage_68": float(np.mean((truth >= predicted[:, 1]) & (truth <= predicted[:, 3]))),
              "coverage_95": float(np.mean((truth >= predicted[:, 0]) & (truth <= predicted[:, 4])))}
    if g0 is not None and parallax is not None:
        distance = 10 ** ((np.asarray(g0) - predicted[:, 2] + 5) / 5)
        relative = distance / (1000 / np.asarray(parallax)) - 1
        result["median_distance_error"] = float(np.median(relative))
        result["nmad_distance_error"] = float(1.4826 * np.median(np.abs(relative - np.median(relative))))
    return result

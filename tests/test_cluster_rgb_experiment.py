"""Check the frozen RGB experiment's sign, grouping and dust propagation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cluster_rgb_experiment", ROOT / "scripts/cluster_rgb_experiment.py")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def test_holdout_is_not_a_development_group():
    assert experiment.HOLDOUT not in experiment.DEVELOPMENT


def test_cluster_weights_and_distance_residual_sign():
    groups = {}
    for name, size in (("first", 20), ("second", 60)):
        color = np.linspace(.45, .75, size)
        groups[name] = {"gr0": color, "rz0": .4*color,
                        "mg": 2.-3.*color}
    model = experiment.fit_model(groups, "A")
    assert model["total_weight_per_system"] == .5
    predicted = experiment.predict(model, groups["first"])
    assert np.median(np.abs(predicted-groups["first"]["mg"])) < .03
    altered = dict(groups["first"])
    altered["mg"] = groups["first"]["mg"]+.2
    # M_g(reference) - M_g(predicted) equals μ(predicted) - μ(reference).
    assert experiment.score(model, altered)["delta_mu_median"] > .15


def test_dust_shift_changes_label_and_both_colors_together():
    group = {"mu": 16., "g0": np.array([17.]), "gr0": np.array([.6]),
             "rz0": np.array([.3]), "mg": np.array([1.])}
    shifted = experiment.shifted(group, debv=.02)
    assert np.isclose(shifted["mg"][0], 1.-3.214*.02)
    assert np.isclose(shifted["gr0"][0], .6-(3.214-2.165)*.02)
    assert np.isclose(shifted["rz0"][0], .3-(2.165-1.211)*.02)

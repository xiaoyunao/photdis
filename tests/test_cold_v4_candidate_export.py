import importlib.util
from pathlib import Path
import sys

import numpy as np


class FixedTree:
    def __init__(self, conflict=False):
        self.conflict = conflict

    def query(self, colors, k):
        distance = np.where(colors[:, 0] < 1.5, 2., .5) if self.conflict else np.full(len(colors), .5)
        return np.repeat(distance[:, None], k, axis=1), np.zeros((len(colors), k), int)


class FixedForest:
    def predict(self, colors):
        return np.tile([7., 7.5, 8., 8.5, 9.], (len(colors), 1))


def test_export_preserves_fallback_reason_and_hypothesis():
    path = Path(__file__).resolve().parents[1] / "scripts/cold_v4_candidate_export.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("candidate_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = {"ls_id": np.array([1, 2]), "source_id": np.array([11, 22]),
             "ra": np.array([1., 2.]), "dec": np.array([0., 0.]),
             "g0": np.array([18., 19.]), "r": np.array([17., 18.]),
             "ebv": np.array([.01, .01]),
             "gr0": np.array([1., 2.]), "rz0": np.array([.5, .5]),
             "zw10": np.array([.2, .2]), "w1w20": np.array([.1, .1]),
             "grz_base": np.ones(2, bool), "grz_w1": np.ones(2, bool),
             "grz_w1_w2": np.ones(2, bool), "grz_clean": np.ones(2, bool),
             "grz_high_quality": np.ones(2, bool),
             "grz_w1_high_quality": np.ones(2, bool),
             "full_high_quality": np.ones(2, bool),
             "valid_w1": np.ones(2, bool), "valid_w2": np.ones(2, bool),
             "w1_clean": np.ones(2, bool), "w2_clean": np.ones(2, bool),
             "snr_w1": np.full(2, 50.), "snr_w2": np.full(2, 40.)}
    route = {"optical_low": np.array([0., 0.]),
             "optical_high": np.array([3., 3.]), "neighbors": 2,
             "tiers": {name: {"tree": FixedTree(name != "A_grz"),
                              "scale": np.ones(len(features)), "radius": 1.}
                       for name, features in module.FEATURES.items()}}
    forests = {name: FixedForest() for name in module.FEATURES}
    out = module.output_rows(table, route, "clean_wise_snr_20", forests)
    assert out["tier"].tolist() == [1, 3]
    assert out["reason_b"].tolist() == [b"support_conflict", b"usable"]
    assert out["infrared_fallback_conflict"].tolist() == [True, False]
    assert out["support_ratio_b"].tolist() == [2., .5]
    assert out["cold_dwarf_hypothesis"].all()
    assert out["depth_state"].tolist() == [b"r_lt_18", b"r18_22_stress_tested"]

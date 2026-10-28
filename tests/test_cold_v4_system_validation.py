import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cold_v4_system_validation import evaluate_pairs


def test_pair_relative_modulus_uses_brightness_and_color_order(tmp_path):
    records = np.array([
        (10, 17.0, .5, 10.0),
        (20, 18.0, .8, 10.2),
        (30, 19.0, .6, 11.0),
    ], dtype=[("source_id", "i8"), ("r0", "f8"),
              ("rz0", "f8"), ("mu_median", "f8")])
    path = tmp_path / "pairs.npz"
    np.savez(path, source_id1=[20, 30], source_id2=[10, 999], group=[10, 30])
    report = evaluate_pairs(records, path)
    assert report["pairs_in_clean_routed_sample"] == 1
    assert report["components"] == 1
    assert np.isclose(report["fainter_minus_brighter_mu"]["median"], .2)
    assert np.isclose(report["redder_minus_bluer_mu"]["median"], .2)

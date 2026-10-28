import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cold_v4_deep_calibration import calibrated_mu, fit_offsets, group_key


def test_split_cqr_uses_independent_sources_and_sparse_group_fallback():
    count = 100
    row = {
        "depth_index": 0, "draws": count, "template_sources": count,
        "selected": np.ones(count, bool),
        "level": np.ones(count, np.uint8),
        "color_group": np.r_[np.zeros(40, int), np.ones(60, int)],
        "error_group": np.zeros(count, np.uint8),
        "support_group": np.zeros(count, np.uint8),
        "truth_mu": np.full(count, 8.0),
        "predicted_mu": np.tile([0., 1., 2., 3., 4.], (count, 1)),
    }
    offsets = fit_offsets([row])
    assert group_key(0, 1) in offsets
    assert group_key(0, 1, 0, 0, 0) not in offsets
    adjusted, applied = calibrated_mu(row, offsets)
    assert applied == {group_key(0, 1): count}
    assert np.all(adjusted[:, 4] >= row["truth_mu"])


def test_repeated_calibration_draws_are_rejected():
    row = {"draws": 4, "template_sources": 2}
    try:
        fit_offsets([row])
    except ValueError as error:
        assert "one draw per source" in str(error)
    else:
        raise AssertionError("Repeated source was treated as independent calibration")

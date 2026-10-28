import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cold_v4_system_validation_v3 import paired_binary_change


def test_pair_route_change_uses_same_systems(tmp_path):
    dtype = [("source_id", "i8"), ("r0", "f8"), ("mu_median", "f8")]
    old = np.array([(1, 17., 10.), (2, 18., 10.1), (3, 20., 11.)], dtype=dtype)
    new = np.array([(1, 17., 10.), (2, 18., 10.2)], dtype=dtype)
    pairs = tmp_path / "pairs.npz"
    np.savez(pairs, source_id1=np.array([1, 2]), source_id2=np.array([2, 3]),
             group=np.array([1, 2]))
    result = paired_binary_change(old, new, pairs)
    assert result["n_pairs"] == 1
    assert np.isclose(result["old_fainter_minus_brighter"]["median"], .1)
    assert np.isclose(result["wise20_fainter_minus_brighter"]["median"], .2)
    assert np.isclose(result["wise20_minus_old_pair_delta"]["median"], .1)

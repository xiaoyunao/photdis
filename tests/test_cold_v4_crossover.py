import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cold_v4_crossover import decompose, nearest_ids


def test_native_degraded_decomposition_is_per_source():
    native = np.array([10., 11.])
    same = np.array([10.2, 11.])
    degraded = np.array([12.4, 13.])
    truth = np.array([9., 10.])
    shift = np.array([2., 2.])
    parts = decompose(native, same, degraded, truth, shift)
    np.testing.assert_allclose(parts["degraded_total"], [1.4, 1.])
    np.testing.assert_allclose(parts["native_residual"], [1., 1.])
    np.testing.assert_allclose(parts["route_change"], [.2, 0.])
    np.testing.assert_allclose(parts["fixed_tier_noise"], [.2, 0.])


def test_target_noise_environments_are_nearest_matched_in_intrinsic_r():
    pool = np.array([(1, 18.1), (2, 18.5), (3, 18.9)],
                    dtype=[("ls_id", "i8"), ("r0", "f8")])
    ids, values = nearest_ids(pool, np.array([18.48, 18.87]))
    np.testing.assert_array_equal(ids, [2, 3])
    np.testing.assert_allclose(values, [18.5, 18.9])

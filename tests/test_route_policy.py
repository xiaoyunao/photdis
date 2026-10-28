import numpy as np
import pytest

from photdis.route_policy import choose_policy


def test_quality_and_wise_policies_preserve_supported_fallback():
    table = {
        "grz_clean": np.array([True, False, True, True]),
        "grz_high_quality": np.array([True, False, True, False]),
        "grz_w1_high_quality": np.array([True, False, True, False]),
        "full_high_quality": np.array([False, False, True, False]),
        "snr_w1": np.array([12., 30., 25., 8.]),
        "snr_w2": np.array([4., 30., 18., 4.]),
    }
    routed = {"tier": np.array([3, 3, 3, 2]),
              "support_by_tier": {
                  "A_grz": np.ones(4, bool),
                  "B_grzW1": np.ones(4, bool),
                  "C_grzW1W2": np.array([True, True, True, False]),
              }}
    assert choose_policy(table, routed, "available").tolist() == [3, 3, 3, 2]
    assert choose_policy(table, routed, "clean").tolist() == [3, 0, 3, 2]
    assert choose_policy(table, routed, "strict").tolist() == [2, 0, 3, 0]
    assert choose_policy(table, routed, "clean_wise_snr_10").tolist() == [2, 0, 3, 1]
    assert choose_policy(table, routed, "clean_wise_snr_20").tolist() == [1, 0, 2, 1]
    with pytest.raises(ValueError):
        choose_policy(table, routed, "unknown")

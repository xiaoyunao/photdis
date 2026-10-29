import numpy as np

from photdis.route_audit import higher_tier_reasons


def test_reliable_infrared_outside_support_is_distinct_from_missing():
    table = {"gr0": np.ones(4), "valid_w1": np.array([1, 0, 1, 1], bool),
             "valid_w2": np.ones(4, bool), "snr_w1": np.array([150, np.nan, 5, 120]),
             "snr_w2": np.array([100, 50, 50, 70]),
             "w1_clean": np.array([1, 0, 1, 1], bool),
             "w2_clean": np.array([1, 1, 1, 0], bool)}
    routed = {"support_by_tier": {"B_grzW1": np.array([0, 0, 0, 1], bool),
                                  "C_grzW1W2": np.zeros(4, bool)}}
    result = higher_tier_reasons(table, routed, np.array([1, 1, 1, 2]))
    assert result["B"].tolist() == ["support_conflict", "missing_or_invalid", "low_snr", "usable"]
    assert result["C"][3] == "quality_failed"
    assert result["higher_support_conflict"].tolist() == [True, False, False, False]

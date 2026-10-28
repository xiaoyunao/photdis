import unittest

import numpy as np

from photdis.candidate import flags
from photdis.support import support_radius


class CandidateTests(unittest.TestCase):
    def test_visible_flags_and_tiers(self):
        training = np.column_stack((np.linspace(0, 1, 20), np.zeros(20)))
        tree, radius = support_radius(training, [1, 1], neighbors=10, quantile=1)
        table = np.array([
            (.5, 0., True, True, True, True, 18.),
            (.5, 0., True, True, False, False, 21.),
            (3., 0., True, True, True, True, 18.),
        ], dtype=[("gr0", "f8"), ("rz0", "f8"), ("grz_base", "?"),
                  ("grz_w1", "?"), ("grz_w1_w2", "?"),
                  ("grz_high_quality", "?"), ("r", "f8")])
        rule = {"color_1pct": [0, -.1], "color_99pct": [1, .1],
                "support_scale_mag": [1, 1],
                "support_10th_neighbor_95pct": radius}
        result = flags(table, rule, tree)
        np.testing.assert_array_equal(result["tier"], [3, 2, 0])
        np.testing.assert_array_equal(result["support_ok"], [True, True, False])
        np.testing.assert_array_equal(result["r_ge_20"], [False, True, False])
        np.testing.assert_array_equal(result["type_unverified"], [True, True, True])


if __name__ == "__main__":
    unittest.main()

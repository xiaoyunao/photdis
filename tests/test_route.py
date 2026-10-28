import unittest

import numpy as np

from photdis.route import apply_route_offsets, fit_route, fit_route_offsets, route_rows


class RouteTests(unittest.TestCase):
    def test_feature_matched_support_falls_back_from_c_to_b(self):
        optical = np.column_stack((np.linspace(0, 1, 20), np.zeros(20)))
        training = {
            "A_grz": optical,
            "B_grzW1": np.column_stack((optical, np.zeros(20))),
            "C_grzW1W2": np.column_stack((optical, np.zeros(20), np.zeros(20))),
        }
        route = fit_route(training, {"A_grz": [1, 1],
                                     "B_grzW1": [1, 1, 1],
                                     "C_grzW1W2": [1, 1, 1, 1]}, quantile=1)
        table = np.array([
            (.5, 0., 0., 0., True, True, True),
            (.5, 0., 0., 5., True, True, True),
            (.5, 0., 5., 5., True, True, True),
            (3., 0., 0., 0., True, True, True),
        ], dtype=[("gr0", "f8"), ("rz0", "f8"), ("zw10", "f8"),
                  ("w1w20", "f8"), ("grz_base", "?"),
                  ("grz_w1", "?"), ("grz_w1_w2", "?")])
        result = route_rows(table, route)
        np.testing.assert_array_equal(result["tier"], [3, 2, 1, 0])
        np.testing.assert_array_equal(result["selected"], [True, True, True, False])

    def test_sparse_calibration_group_uses_global_offset(self):
        truth = np.zeros(12)
        prediction = np.tile([-1, -.5, 0, .5, 1], (12, 1)).astype(float)
        truth[-1] = 3
        groups = np.array([0] * 11 + [5])
        offsets = fit_route_offsets(truth, prediction, groups, min_group=2)
        self.assertEqual(offsets["5"]["source"], "global")
        adjusted = apply_route_offsets(prediction, groups, offsets)
        self.assertTrue(np.all(adjusted[:, 0] <= prediction[:, 0]))
        self.assertTrue(np.all(adjusted[:, 4] >= prediction[:, 4]))


if __name__ == "__main__":
    unittest.main()

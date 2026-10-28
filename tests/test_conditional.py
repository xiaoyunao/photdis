import unittest

import numpy as np

from photdis.conditional import (apply_group_cqr, correct_center,
                                  fit_center, fit_group_cqr, group_labels)


class ConditionalTests(unittest.TestCase):
    def test_center_subtracts_predicted_minus_truth_offset(self):
        x = np.column_stack((np.linspace(-1, 1, 120), np.linspace(0, 1, 120)))
        residual = np.full(120, .2)
        scaler, model = fit_center(x, residual)
        q = np.tile([-1, -.5, 0, .5, 1], (120, 1))
        corrected = correct_center(q, x, scaler, model)
        np.testing.assert_allclose(corrected[:, 2], -.2, atol=1e-10)

    def test_fixed_groups_and_fallback_expand_intervals(self):
        rz = np.array([0.1, .5, .9])
        r = np.array([16, 18, 19])
        np.testing.assert_array_equal(group_labels(rz, r, [0, .3, .7, 1], [17, 18, 18]),
                                      [0, 3, 5])
        truth = np.linspace(-1, 1, 120)
        prediction = np.tile([-.2, -.1, 0, .1, .2], (120, 1))
        groups = np.zeros(120, int)
        offsets = fit_group_cqr(truth, prediction, groups, min_group=80)
        self.assertEqual(offsets["0"]["source"], "color_x_depth")
        self.assertEqual(offsets["1"]["source"], "color_only")
        adjusted = apply_group_cqr(prediction, groups, offsets)
        self.assertTrue(np.all(adjusted[:, 0] <= prediction[:, 0]))
        self.assertTrue(np.all(adjusted[:, 4] >= prediction[:, 4]))


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np

from photdis.followup import apply_cqr, cqr_offset, identity_hash, ps1_replacement


class FollowupTests(unittest.TestCase):
    def test_identity_hash_is_order_independent_and_unique(self):
        self.assertEqual(identity_hash([8, 2, 5]), identity_hash([5, 8, 2]))
        with self.assertRaises(ValueError):
            identity_hash([2, 2])

    def test_cqr_uses_finite_sample_rank(self):
        pred = np.column_stack([np.zeros(19)] * 5)
        truth = np.arange(19, dtype=float)
        offset, n = cqr_offset(truth, pred, 0, 4, .05)
        self.assertEqual(n, 19)
        self.assertEqual(offset, 18)
        new = apply_cqr(pred, 3, offset)
        self.assertEqual(new[0, 0], -18)
        self.assertEqual(new[0, 4], 18)

    def test_ps1_replaces_colors_and_label_together(self):
        t = {"ps1_g_decam": np.array([17.0]), "ps1_r_decam": np.array([16.0]),
             "ps1_z_decam": np.array([15.5]), "g0": np.array([16.9]),
             "r0": np.array([15.9]), "z0": np.array([15.4]),
             "g": np.array([17.1]), "r": np.array([16.1]), "z": np.array([15.6]),
             "p_corr": np.array([10.0])}
        colors, mg = ps1_replacement(t)
        np.testing.assert_allclose(colors, [[1.0, .5]])
        np.testing.assert_allclose(mg, [11.8])


if __name__ == "__main__":
    unittest.main()

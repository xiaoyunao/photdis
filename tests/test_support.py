import unittest

import numpy as np

from photdis.support import support_radius, supported


class SupportTests(unittest.TestCase):
    def test_training_excludes_self_but_external_query_uses_tenth(self):
        training = np.arange(11, dtype=float)[:, None]
        tree, radius = support_radius(training, [1], neighbors=10, quantile=1)
        self.assertEqual(radius, 10)
        self.assertTrue(supported(tree, [[0.0]], [1], radius, neighbors=10)[0])
        self.assertFalse(supported(tree, [[21.0]], [1], radius, neighbors=10)[0])

    def test_requires_enough_finite_training_points(self):
        with self.assertRaises(ValueError):
            support_radius(np.arange(10)[:, None], [1], neighbors=10)
        with self.assertRaises(ValueError):
            support_radius(np.array([[0.0], [np.nan]]), [1], neighbors=1)


if __name__ == "__main__":
    unittest.main()

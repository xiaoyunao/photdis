import unittest

import numpy as np

from photdis.quantile import QUANTILES, QuantileForest, metrics, spatial_partition


class QuantileTests(unittest.TestCase):
    def test_leaf_distribution_includes_response_scatter(self):
        x = np.arange(80, dtype=float)[:, None]
        y = np.tile([-2.0, 2.0], 40)
        model = QuantileForest(n_estimators=20, min_samples_leaf=40).fit(x, y)
        result = model.predict(np.array([[25.0], [55.0]]))
        self.assertEqual(result.shape, (2, len(QUANTILES)))
        self.assertTrue(np.all(np.diff(result, axis=1) >= 0))
        self.assertTrue(np.all(np.isin(result, [-2.0, 2.0])))
        self.assertLess(result[0, 0], -1)
        self.assertGreater(result[0, -1], 1)

    def test_spatial_cells_never_split(self):
        cells = np.repeat(np.arange(70), 10)
        selection = np.ones(len(cells), bool)
        partition, mapping = spatial_partition(cells, selection)
        np.testing.assert_array_equal(partition, np.repeat(partition[::10], 10))
        self.assertEqual(len(mapping), 70)
        self.assertTrue(np.all(np.bincount(partition, minlength=4)[:3] >= 100))

    def test_interval_metrics(self):
        truth = np.array([0.0, 1.0, 2.0])
        pred = np.column_stack([truth - 2, truth - 1, truth, truth + 1, truth + 2])
        result = metrics(truth, pred)
        self.assertEqual(result["coverage_68"], 1.0)
        self.assertEqual(result["coverage_95"], 1.0)
        self.assertEqual(result["nmad_mag"], 0.0)


if __name__ == "__main__":
    unittest.main()

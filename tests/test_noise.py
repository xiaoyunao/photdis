import unittest

import numpy as np

from photdis.noise import (simulate_shared_flux, transformed_distance_quantiles,
                           interval_diagnostics)
from photdis.stage1 import PHOT_COLUMNS


class NoiseTests(unittest.TestCase):
    def test_shared_flux_shift_and_noise_budget(self):
        names = ["maskbits", *[name.lower() for name in PHOT_COLUMNS]]
        dtype = [(name, "f8") for name in names] + [("r0", "f8")]
        source = np.zeros(1, dtype=dtype)
        target = np.zeros(1, dtype=dtype)
        source["r0"] = 17.5
        target["maskbits"] = 0
        for band in ("g", "r", "z", "w1", "w2"):
            source[f"flux_{band}"] = 100
            source[f"flux_ivar_{band}"] = 100
            source[f"mw_transmission_{band}"] = 1
            target[f"flux_ivar_{band}"] = 100 * (10 ** .4) ** 2
            target[f"mw_transmission_{band}"] = 1
            target[f"nobs_{band}"] = 1
        target["flux_r"] = 100 / (10 ** .4)
        raw, shift, ratio = simulate_shared_flux(source, target, np.random.default_rng(3))
        self.assertAlmostEqual(shift[0], 1)
        self.assertAlmostEqual(raw["FLUX_G"][0], target["flux_r"][0])
        self.assertAlmostEqual(ratio[0, 0], 1)

    def test_absolute_magnitude_to_distance_quantiles(self):
        mu, distance = transformed_distance_quantiles([20.0], [[5, 6, 7, 8, 9]])
        np.testing.assert_allclose(mu, [[11, 12, 13, 14, 15]])
        np.testing.assert_allclose(distance[0, 2], 10 ** 3.6)
        diagnostic = interval_diagnostics([13], mu)
        self.assertEqual(diagnostic["coverage_68"], 1)
        self.assertEqual(diagnostic["coverage_95"], 1)

    def test_template_noise_sets_effective_inverse_variance(self):
        names = ["maskbits", *[name.lower() for name in PHOT_COLUMNS]]
        dtype = [(name, "f8") for name in names] + [("r0", "f8")]
        source = np.zeros(1, dtype=dtype)
        target = np.zeros(1, dtype=dtype)
        source["r0"] = 17.5
        target["flux_r"] = 100
        for band in ("g", "r", "z", "w1", "w2"):
            source[f"flux_{band}"] = 100
            source[f"flux_ivar_{band}"] = 1
            source[f"mw_transmission_{band}"] = 1
            target[f"flux_ivar_{band}"] = 100
            target[f"mw_transmission_{band}"] = 1
            target[f"nobs_{band}"] = 1
        raw, _, ratio = simulate_shared_flux(source, target, np.random.default_rng(3))
        self.assertAlmostEqual(ratio[0, 0], 10)
        self.assertAlmostEqual(raw["FLUX_IVAR_G"][0], 1)
        self.assertAlmostEqual(raw["FLUX_G"][0], 100)


if __name__ == "__main__":
    unittest.main()

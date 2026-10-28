import unittest

import numpy as np

from photdis.noise import (simulate_shared_flux, transformed_distance_quantiles,
                           interval_diagnostics)
from photdis.photometry import derive_photometry
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

    def test_cross_field_dust_transmission_preserves_intrinsic_colors(self):
        names = ["maskbits", *[name.lower() for name in PHOT_COLUMNS]]
        dtype = [(name, "f8") for name in names] + [("r0", "f8")]
        source = np.zeros(1, dtype=dtype)
        target = np.zeros(1, dtype=dtype)
        source["r0"] = 15.0
        source_trans = dict(g=.72, r=.80, z=.90, w1=.97, w2=.98)
        target_trans = dict(g=.88, r=.92, z=.96, w1=.99, w2=.995)
        for band in source_trans:
            source[f"flux_{band}"] = 1000 * source_trans[band]
            source[f"flux_ivar_{band}"] = 1
            source[f"mw_transmission_{band}"] = source_trans[band]
            target[f"flux_{band}"] = 100 * target_trans[band]
            target[f"flux_ivar_{band}"] = 1e12
            target[f"mw_transmission_{band}"] = target_trans[band]
            target[f"nobs_{band}"] = 1
        raw, shift, _ = simulate_shared_flux(source, target, np.random.default_rng(8))
        phot = derive_photometry(raw)
        self.assertAlmostEqual(float(shift[0]), 2.5, places=5)
        for band in source_trans:
            self.assertAlmostEqual(float(phot[f"{band}0"][0]), 17.5, places=5)
        for color in ("gr0", "rz0", "zw10", "w1w20"):
            self.assertAlmostEqual(float(phot[color][0]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

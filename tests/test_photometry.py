"""Scientific invariants and selection boundaries for measured photometry."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photdis.photometry import BANDS, MAG_ERROR_FACTOR, derive_photometry


def catalog(size=1):
    fields = [("MASKBITS", "i4")]
    for band in BANDS:
        fields.extend((f"{key}_{band}", "f4") for key in
                      ("FLUX", "FLUX_IVAR", "MW_TRANSMISSION", "FRACFLUX"))
        fields.append((f"NOBS_{band}", "i2"))
    for band in ("G", "R", "Z"):
        fields.extend((f"{key}_{band}", "f4") for key in
                      ("FRACMASKED", "FRACIN", "RCHISQ"))
        fields.append((f"ANYMASK_{band}", "i2"))
    fields.extend((f"WISEMASK_{band}", "u1") for band in ("W1", "W2"))
    data = np.zeros(size, dtype=fields)
    for band in BANDS:
        data[f"FLUX_{band}"] = 100.0
        data[f"FLUX_IVAR_{band}"] = 100.0
        data[f"MW_TRANSMISSION_{band}"] = 1.0
        data[f"NOBS_{band}"] = 2
    for band in ("G", "R", "Z"):
        data[f"FRACIN_{band}"] = 1.0
    return data


class PhotometryTests(unittest.TestCase):
    def test_units_extinction_and_common_dimming(self):
        data = catalog()
        for band in BANDS:
            data[f"MW_TRANSMISSION_{band}"] = 0.5
        out = derive_photometry(data)
        self.assertAlmostEqual(float(out["g"][0]), 17.5, places=6)
        self.assertAlmostEqual(float(out["g0"][0]),
                               17.5 + 2.5 * np.log10(0.5), places=5)
        self.assertAlmostEqual(float(out["sigma_g"][0]),
                               MAG_ERROR_FACTOR / 1000, places=8)
        dimmed = data.copy()
        for band in BANDS:
            dimmed[f"FLUX_{band}"] /= 2.0
        after = derive_photometry(dimmed)
        for color in ("gr0", "rz0", "zw10", "w1w20"):
            np.testing.assert_allclose(after[color], out[color], atol=1e-6)
        np.testing.assert_allclose(after["g0"] - out["g0"],
                                   2.5 * np.log10(2), atol=1e-6)

    def test_optional_ir_and_invalid_measured_flux(self):
        data = catalog(6)
        data["FLUX_W2"][1] = -1.0
        data["FLUX_IVAR_W1"][2] = 0.0
        data["FLUX_IVAR_G"][3] = 0.0
        data["MW_TRANSMISSION_R"][4] = 0.0
        data["FLUX_G"][5] = np.nan
        original = data.copy()
        with np.errstate(all="raise"):
            out = derive_photometry(data)
        np.testing.assert_array_equal(out["grz_base"],
                                      [True, True, True, False, False, False])
        np.testing.assert_array_equal(out["photometry_tier"], [2, 1, 0, -1, -1, -1])
        self.assertTrue(np.isnan(out["w20"][1]))
        self.assertTrue(np.isnan(out["sigma_g"][3]))
        for name in data.dtype.names:
            np.testing.assert_equal(data[name], original[name])

    def test_clean_subset_preserves_parent_and_diagnostic_masks(self):
        data = catalog(6)
        data["ANYMASK_G"][1] = 2
        data["RCHISQ_R"][1] = 100.0
        data["MASKBITS"][2] = (1 << 1) | (1 << 11) | (1 << 12) | (1 << 13)
        data["MASKBITS"][3] = 1 << 4
        data["FRACIN_R"][4] = 0.2
        data["FRACFLUX_Z"][5] = 0.6
        out = derive_photometry(data)
        self.assertTrue(out["grz_base"].all())
        self.assertTrue(out["grz_precise"].all())
        np.testing.assert_array_equal(out["grz_high_quality"],
                                      [True, True, True, False, False, False])

    def test_ir_quality_does_not_change_optical_parent(self):
        data = catalog(3)
        data["WISEMASK_W2"][0] = 1
        data["FRACFLUX_W1"][1] = 2.0
        data["FLUX_IVAR_W2"][2] = (5.0 / 100.0) ** 2
        out = derive_photometry(data)
        self.assertTrue(out["grz_high_quality"].all())
        np.testing.assert_array_equal(out["photometry_tier"], [1, 0, 2])
        self.assertFalse(out["full_high_quality"].any())

    def test_gr_quality_uses_only_its_input_bands(self):
        data = catalog(5)
        data["FLUX_IVAR_Z"][0] = 0
        data["MASKBITS"][1] = 1 << 4  # z saturation
        data["FRACMASKED_Z"][2] = 0.8
        data["MASKBITS"][3] = 1 << 2  # g saturation
        data["FRACIN_R"][4] = 0.1
        out = derive_photometry(data)
        np.testing.assert_array_equal(out["gr_clean"],
                                      [True, True, True, False, False])
        self.assertFalse(out["grz_clean"].any())


if __name__ == "__main__":
    unittest.main()

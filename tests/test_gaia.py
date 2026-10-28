"""Exact ID matching and signed-parallax/calibration-domain invariants."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import fitsio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photdis import gaia as module


def good_gaia(size):
    result = module._empty_columns(size)
    result["gaia_match"][:] = True
    result["parallax"][:] = 1.0
    result["parallax_error"][:] = 0.02
    result["ruwe"][:] = 1.0
    result["visibility_periods_used"][:] = 12
    result["astrometric_params_solved"][:] = 31
    result["phot_g_mean_mag"][:] = 15.0
    result["nu_eff_used_in_astrometry"][:] = 1.5
    result["ecl_lat"][:] = 20.0
    return result


def write_tile(root, pixel, source_ids):
    columns = good_gaia(len(source_ids))
    fields = [(name, columns[name.lower()].dtype) for name in module.GAIA_COLUMNS]
    table = np.empty(len(source_ids), dtype=fields)
    for name in module.GAIA_COLUMNS:
        table[name] = columns[name.lower()]
    table["SOURCE_ID"] = source_ids
    table["PARALLAX"] = np.arange(len(table)) - 1.0
    # Test real source-file byte codes rather than only int16 fixture codes.
    dtype = [(name, "i1" if name == "ASTROMETRIC_PARAMS_SOLVED" else dt)
             for name, dt in fields]
    table = table.astype(dtype)
    fitsio.write(str(Path(root) / f"healpix-{pixel:05d}.fits"), table,
                 extname="GAIAHPX", header={"HPXNSIDE": 32, "HPXNEST": True})


class FakeZeroPoint:
    def __init__(self):
        self.calls = []

    def get_zpt(self, g, nu, pseudo, latitude, solved, *, _warnings):
        self.calls.append((g.copy(), solved.copy(), _warnings))
        result = np.full(len(g), -0.02)
        result[g == 10] = np.nan
        return result


class GaiaTests(unittest.TestCase):
    def setUp(self):
        module._read_healpix.cache_clear()

    def test_exact_ids_order_repeats_missing_and_bounded_cache(self):
        unit = module.SOURCE_ID_HEALPIX_FACTOR
        with tempfile.TemporaryDirectory() as root:
            write_tile(root, 0, [100, 30])
            write_tile(root, 1, [unit + 7])
            query = np.array([unit + 7, 30, 0, 100, 31, 30, 2 * unit + 9], dtype=np.int64)
            result = module.join_gaia(query, root)
            np.testing.assert_array_equal(result["gaia_match"],
                                          [True, True, False, True, False, True, False])
            np.testing.assert_array_equal(result["gaia_file_missing"],
                                          [False, False, False, False, False, False, True])
            np.testing.assert_array_equal(result["source_id"],
                                          [unit + 7, 30, -1, 100, -1, 30, -1])
            np.testing.assert_allclose(result["parallax"][[0, 1, 3, 5]], [-1, 0, -1, 0])
            self.assertTrue(np.isnan(result["parallax"][[2, 4, 6]]).all())
            self.assertEqual(result["astrometric_params_solved"].dtype, np.dtype("int16"))
            before = module._read_healpix.cache_info()
            module.join_gaia(query, root)
            after = module._read_healpix.cache_info()
            self.assertEqual(after.misses, before.misses)
            self.assertEqual(after.hits - before.hits, 3)
            self.assertEqual(after.maxsize, 4)

    def test_no_float_identity_coercion_or_duplicate_catalog_identity(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "integer array"):
                module.join_gaia(np.array([float(2**60)]), root)
            write_tile(root, 0, [30, 30])
            with self.assertRaisesRegex(ValueError, "Duplicate Gaia"):
                module.join_gaia(np.array([30], dtype=np.int64), root)

    def test_signed_parallaxes_quality_and_calibration_domain(self):
        data = good_gaia(13)
        data["parallax"][:3] = [-0.2, 0.0, 0.4]
        data["parallax_error"][:2] = 0.1
        data["duplicated_source"][3] = True
        data["visibility_periods_used"][4] = 8
        data["phot_g_mean_mag"][5] = 21
        data["astrometric_params_solved"][[6, 7]] = 95
        data["nu_eff_used_in_astrometry"][[6, 7]] = np.nan
        data["pseudocolour"][[6, 7]] = [1.5, 2.0]
        data["parallax_error"][8] = 0
        data["in_qso_candidates"][9] = True
        data["astrometric_params_solved"][10] = 3
        data["gaia_match"][11] = False
        data["phot_g_mean_mag"][12] = 10
        original_parallax = data["parallax"].copy()
        fake = FakeZeroPoint()
        with patch.object(module, "_get_zero_point", return_value=fake):
            out = module.derive_gaia_labels(data)
        self.assertTrue(out["astro_quality"][0])
        self.assertTrue(out["negative_raw_parallax"][0])
        self.assertTrue(out["negative_corrected_parallax"][0])
        self.assertAlmostEqual(float(out["p_corr"][0]), -0.18, places=6)
        self.assertAlmostEqual(float(out["p_corr"][1]), 0.02, places=6)
        self.assertTrue(out["anchor_astrometry"][2])
        self.assertFalse(out["astro_quality"][[3, 4, 8, 10, 11]].any())
        self.assertFalse(out["zpt_inrange"][[5, 7, 10, 11]].any())
        self.assertTrue(np.isnan(out["p_corr"][[5, 7, 10, 11, 12]]).all())
        self.assertTrue(out["anchor_astrometry"][6])
        self.assertTrue(out["contaminant_candidate"][9])
        self.assertTrue(out["anchor_astrometry"][9])  # Flag is separate, not silently filtered.
        self.assertTrue(out["zpt_inrange"][12])
        self.assertFalse(out["zpt_valid"][12])
        self.assertEqual(len(fake.calls), 1)
        self.assertIs(fake.calls[0][2], False)
        self.assertTrue(np.all((fake.calls[0][0] > 6) & (fake.calls[0][0] < 21)))
        np.testing.assert_equal(data["parallax"], original_parallax)


if __name__ == "__main__":
    unittest.main()

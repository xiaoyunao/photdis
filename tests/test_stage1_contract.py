"""Synthetic FITS checks for stage-one joins, signed labels and empty output."""

from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

import fitsio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photdis import gaia, stage1
from photdis.holdouts import HoldoutReferences


def sweep_rows(size):
    integer = {"RELEASE", "BRICKID", "OBJID", "MASKBITS", "REF_ID", "FITBITS"}
    fields = []
    for name in stage1.CORE_COLUMNS + stage1.PHOT_COLUMNS:
        dtype = "f8"
        if name in {"TYPE", "REF_CAT"}:
            dtype = "S4"
        elif name in integer or name.startswith(("NOBS_", "ANYMASK_", "WISEMASK_")):
            dtype = "i8"
        fields.append((name, dtype))
    data = np.zeros(size, dtype=fields)
    data["RELEASE"], data["BRICKID"], data["OBJID"] = 11000, 100, np.arange(size)
    data["TYPE"], data["REF_CAT"] = b"PSF", b"G3"
    data["RA"], data["DEC"], data["EBV"] = 202.0, 22.0, 0.01
    data["REF_ID"], data["REF_EPOCH"] = np.arange(100, 100 + size), 2016.0
    data["PARALLAX_IVAR"] = 100.0
    for band in stage1.BANDS:
        data[f"FLUX_{band}"] = 100.0
        data[f"FLUX_IVAR_{band}"] = 1.0
        data[f"MW_TRANSMISSION_{band}"] = 1.0
        data[f"NOBS_{band}"] = 2
    for band in ("G", "R", "Z"):
        data[f"FRACIN_{band}"] = 1.0
    return data


def empty_spectroscopy():
    fields = [(name, "f8") for name in
              ("ra", "dec", "snr", "teff", "teff_err", "logg", "logg_err", "feh", "feh_err")]
    fields += [(name, "i8") for name in ("source_id", "object_id", "quality_reason")]
    fields += [(name, "?") for name in
               ("quality", "is_stellar", "known_contaminant", "label_dwarf", "label_giant")]
    fields += [("spectype", "S8")]
    return {"table": np.zeros(0, dtype=fields), "ids": np.zeros(0, dtype="i8"),
            "rows": np.zeros(0, dtype="i8")}


def references():
    columns = {name: np.zeros(0, dtype="i8" if "group" in name or "source_id" in name else "f8")
               for name in HoldoutReferences.__dataclass_fields__}
    columns["member_source_id"] = np.array([102], dtype="i8")
    columns["member_group"] = np.array([1], dtype="i8")
    return HoldoutReferences(**columns)


def write_gaia(root):
    columns = gaia._empty_columns(3)
    columns.update(source_id=np.array([100, 101, 102], dtype="i8"),
                   ra=np.full(3, 202.), dec=np.full(3, 22.),
                   parallax=np.array([-0.2, 0.0, 1.0], dtype="f4"),
                   parallax_error=np.array([0.1, 0.1, 0.02], dtype="f4"),
                   ruwe=np.ones(3, dtype="f4"),
                   visibility_periods_used=np.full(3, 12, dtype="i2"),
                   astrometric_params_solved=np.full(3, 31, dtype="i2"),
                   phot_g_mean_mag=np.full(3, 15., dtype="f4"),
                   nu_eff_used_in_astrometry=np.full(3, 1.5, dtype="f4"),
                   ecl_lat=np.full(3, 20.))
    table = stage1.structured({name: columns[name.lower()] for name in gaia.GAIA_COLUMNS})
    fitsio.write(str(root / "healpix-00000.fits"), table,
                 header={"HPXNSIDE": 32, "HPXNEST": True})


class ZeroPoint:
    def get_zpt(self, g, nu, pseudo, lat, solved, *, _warnings):
        return np.full(len(g), -0.02)


class Stage1ContractTests(unittest.TestCase):
    def setUp(self):
        gaia._read_healpix.cache_clear()

    def run_tile(self, data, root):
        path = root / "sweep-200p020-205p025.fits"
        fitsio.write(str(path), data)
        gaia_root = root / "gaia"
        gaia_root.mkdir()
        write_gaia(gaia_root)
        config = {"sky": {"abs_b_min": 30, "ebv_max": 0.03}, "photometry": {},
                  "gaia_root": str(gaia_root), "spectroscopy_match_radius_arcsec": 1.,
                  "coverage_reference_modulus": 1, "split": {"nside": 8}}
        context = {"config": config, "run_dir": root / "output", "signature": "test",
                   "references": references(), "partitions": np.zeros(768, dtype="i2"),
                   "spectroscopy": {name: empty_spectroscopy() for name in ("desi", "apogee")}}
        with patch.dict(stage1.CTX, context, clear=True), patch.object(gaia, "_get_zero_point", return_value=ZeroPoint()):
            metadata = stage1.process_tile(str(path))
        return fitsio.read(metadata["output"], ext=1), metadata

    def test_signed_parallaxes_remain_extended_and_external_anchor_is_reserved(self):
        with tempfile.TemporaryDirectory() as root:
            table, metadata = self.run_tile(sweep_rows(3), Path(root))
        self.assertEqual(len(table), 3)
        self.assertTrue(table["extended"].all())
        self.assertTrue(table["extended_quality"].all())
        np.testing.assert_allclose(table["gaia_parallax"], [-0.2, 0.0, 1.0])
        np.testing.assert_allclose(table["p_corr"], [-0.18, 0.02, 1.02], atol=1e-7)
        np.testing.assert_array_equal(table["anchor"], [False, False, True])
        np.testing.assert_array_equal(table["partition"], [0, 0, 3])
        np.testing.assert_array_equal(table["train_eligible"], [True, True, False])
        self.assertTrue(np.isnan(table["mg"][:2]).all())
        self.assertEqual(metadata["counts"]["negative_parallax"], 1)
        self.assertEqual(metadata["counts"]["zero_parallax"], 1)

    def test_empty_input_and_all_rows_rejected_write_readable_empty_fits(self):
        for size in (0, 2):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as root:
                data = sweep_rows(size)
                data["EBV"] = 0.5
                table, metadata = self.run_tile(data, Path(root))
                self.assertEqual(len(table), 0)
                self.assertEqual(metadata["rows"], 0)
                self.assertIn("gaia_parallax", table.dtype.names)
                self.assertIn("extended", table.dtype.names)
                run_dir = Path(root) / "output"
                stage1.write_json(run_dir / "scan_manifest.json", {
                    "signature": "test", "files": [metadata["source"]],
                    "code_commit": "synthetic", "all_source_tiles": 1,
                    "source_bytes": metadata["source_bytes"], "limited_smoke_run": True,
                })
                with patch.object(stage1, "require_compute"), patch.object(stage1, "run_signature", return_value="test"), contextlib.redirect_stdout(io.StringIO()):
                    stage1.finalize({"output_dir": str(run_dir), "coverage_reference_modulus": 1})
                for role in stage1.ROLE_NAMES:
                    self.assertEqual(len(fitsio.read(str(run_dir / "tables" / f"{role}.fits"), ext=1)), 0)


if __name__ == "__main__":
    unittest.main()

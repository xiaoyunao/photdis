"""Leakage regressions across catalog identity recovery and final deduplication."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import fitsio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from photdis import stage1
from photdis.gaia import _empty_columns
from photdis.holdouts import HoldoutReferences, annotate_holdouts
from photdis.spectroscopy import _COMMON_DTYPE


KNOWN_ID = 9007199254740993


def references():
    return HoldoutReferences(
        np.array([KNOWN_ID + 2], dtype="i8"), np.array([1], dtype="i8"),
        np.array([KNOWN_ID, KNOWN_ID + 1], dtype="i8"), np.array([KNOWN_ID, KNOWN_ID], dtype="i8"),
        np.array([359.9]), np.array([0.]), np.array([0.5]), np.array([100001], dtype="i8"))


def optical_rows(size=1, source_id=0):
    types = {"TYPE": "S4", "REF_CAT": "S2", "REF_ID": "i8", "RA": "f8", "DEC": "f8"}
    integer = {"RELEASE", "BRICKID", "OBJID", "MASKBITS", "FITBITS"}
    names = stage1.CORE_COLUMNS + stage1.PHOT_COLUMNS
    table = np.zeros(size, dtype=[(name, types.get(name, "i4" if name in integer else "f4")) for name in names])
    table["RELEASE"], table["BRICKID"], table["OBJID"] = 11000, 42, np.arange(size) + 1
    table["TYPE"], table["RA"], table["DEC"], table["EBV"] = b"PSF", 180.1 + np.arange(size) / 36000, 1., 0.001
    table["REF_CAT"], table["REF_ID"] = b"G3", source_id
    for band in stage1.BANDS:
        table[f"FLUX_{band}"] = table[f"FLUX_IVAR_{band}"] = 100
        table[f"MW_TRANSMISSION_{band}"] = 1
        table[f"NOBS_{band}"] = 2
    for band in ("G", "R", "Z"):
        table[f"FRACIN_{band}"] = 1
    return table


def spectrum(source_id=0, object_id=b"SPECTRUM", count=1):
    table = np.zeros(count, dtype=_COMMON_DTYPE)
    table["source_id"], table["ra"], table["dec"] = source_id, 180.1, 1.
    table["quality"], table["is_stellar"], table["object_id"] = True, True, object_id
    table["snr"], table["teff"], table["logg"], table["feh"] = 50., 5000., 4.5, -0.5
    rows = np.flatnonzero(table["source_id"] > 0)
    return {"table": table, "ids": table["source_id"][rows], "rows": rows}


class Stage1HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.old_context = stage1.CTX.copy()
        self.addCleanup(self.restore_context)

    def restore_context(self):
        stage1.CTX.clear()
        stage1.CTX.update(self.old_context)

    def process(self, raw, desi=None, apogee=None):
        source = self.root / "sweep-180p000-190p005.fits"
        fitsio.write(str(source), raw)
        config = {"sky": {"abs_b_min": 0., "ebv_max": 1.}, "photometry": {},
                  "gaia_root": "unused", "spectroscopy_match_radius_arcsec": 1.,
                  "coverage_reference_modulus": 1, "split": {"nside": 1},
                  "output_dir": str(self.root)}
        stage1.CTX.clear()
        stage1.CTX.update(config=config, run_dir=self.root, signature="regression",
                          references=references(), partitions=np.zeros(12, dtype="i2"),
                          spectroscopy={"desi": desi or spectrum(count=0), "apogee": apogee or spectrum(count=0)})
        # Simulate an unavailable full Gaia record. Known IDs must still reserve
        # objects, and recovered IDs must never pass through floating point.
        with patch.object(stage1, "join_gaia", side_effect=lambda ids, root: _empty_columns(len(ids))):
            metadata = stage1.process_tile(str(source))
        return config, source, metadata, fitsio.read(metadata["output"], ext=1)

    def finalize(self, config, source, metadata):
        manifest = {"signature": "regression", "files": [str(source)], "code_commit": "fixture",
                    "all_source_tiles": 1, "source_bytes": source.stat().st_size, "limited_smoke_run": True}
        (self.root / "scan_manifest.json").write_text(json.dumps(manifest))
        with patch.object(stage1, "require_compute"), patch.object(stage1, "run_signature", return_value="regression"), redirect_stdout(io.StringIO()):
            stage1.finalize(config)
        return fitsio.read(str(self.root / "tables/parent.fits"), ext=1)

    def test_exact_id_and_ra_wrap_masks(self):
        result = annotate_holdouts(references(), np.array([KNOWN_ID, KNOWN_ID + 2, 0], dtype="i8"),
                                   np.array([180., 180., 0.1]), np.zeros(3))
        np.testing.assert_array_equal(result["widebinary"], [True, False, False])
        np.testing.assert_array_equal(result["external_cluster_member"], [False, True, False])
        np.testing.assert_array_equal(result["cluster_region"], [False, False, True])
        self.assertTrue(result["external_any"].all())
        with self.assertRaises(TypeError):
            annotate_holdouts(references(), np.array([float(KNOWN_ID)]), np.array([0.]), np.array([0.]))

    def test_known_external_identity_without_full_gaia_record(self):
        _, _, _, table = self.process(optical_rows(source_id=KNOWN_ID))
        self.assertEqual(int(table["source_id"][0]), KNOWN_ID)
        self.assertEqual(int(table["partition"][0]), 3)
        self.assertFalse(table["train_eligible"][0])

    def test_spectroscopic_identity_recovery_precedes_holdouts(self):
        _, _, _, table = self.process(optical_rows(), desi=spectrum(KNOWN_ID))
        self.assertTrue(table["gaia_identity_recovered"][0])
        self.assertEqual(int(table["source_id"][0]), KNOWN_ID)
        self.assertTrue(table["widebinary"][0])
        self.assertEqual(int(table["partition"][0]), 3)
        self.assertFalse(table["train_eligible"][0])

    def test_conflicting_survey_identities_are_reserved(self):
        _, _, _, table = self.process(optical_rows(), desi=spectrum(KNOWN_ID), apogee=spectrum(KNOWN_ID + 1))
        self.assertFalse(table["gaia_identity_recovered"][0])
        self.assertTrue(table["identity_ambiguity_reserved"][0])
        self.assertEqual(int(table["partition"][0]), 3)
        self.assertFalse(table["train_eligible"][0])

    def test_shared_spectrum_cannot_cross_spatial_partitions(self):
        config, source, metadata, table = self.process(optical_rows(2), desi=spectrum())
        # Simulate two retained optical counterparts straddling a split boundary.
        table["partition"] = [0, 2]
        table["train_eligible"] = [True, False]
        fitsio.write(metadata["output"], table, clobber=True)
        parent = self.finalize(config, source, metadata)
        self.assertEqual(len(parent), 2)
        self.assertTrue(parent["identity_ambiguity_reserved"].all())
        self.assertTrue((parent["partition"] == 3).all())
        self.assertFalse(parent["train_eligible"].any())
        self.assertFalse(parent["external_any"].any())  # distinct reason flags

    def test_global_gaia_identity_dedup_keeps_one_external_source(self):
        config, source, metadata, _ = self.process(optical_rows(2, source_id=KNOWN_ID))
        parent = self.finalize(config, source, metadata)
        self.assertEqual(len(parent), 1)
        self.assertEqual(int(parent["source_id"][0]), KNOWN_ID)
        self.assertEqual(int(parent["partition"][0]), 3)
        self.assertFalse(parent["train_eligible"][0])

    def test_discarded_duplicate_must_not_erase_cross_identity_spectrum_link(self):
        # A has two optical detections associated with spectra X and Y. B has
        # spectrum Y too. Choosing A/X as the representative cannot erase the
        # A--Y--B identity ambiguity and allow B into training beside external A.
        raw = optical_rows(3, source_id=np.array([KNOWN_ID, KNOWN_ID, KNOWN_ID + 100], dtype="i8"))
        config, source, metadata, table = self.process(raw)
        table["desi_object_id"] = [b"X", b"Y", b"Y"]
        for name in ("desi_match", "desi_quality", "spectroscopy_match", "spectro_quality"):
            table[name] = True
        table["train_eligible"][-1] = True
        fitsio.write(metadata["output"], table, clobber=True)
        parent = self.finalize(config, source, metadata)
        self.assertEqual(len(parent), 2)
        self.assertTrue((parent["partition"] == 3).all())
        self.assertFalse(parent["train_eligible"].any())

    def check_discarded_reservation_survives(self, flag):
        config, source, metadata, table = self.process(optical_rows(2, source_id=KNOWN_ID + 100))
        table[flag][-1] = True
        table["partition"] = [0, 3]
        table["train_eligible"] = [True, False]
        fitsio.write(metadata["output"], table, clobber=True)
        parent = self.finalize(config, source, metadata)
        self.assertEqual(len(parent), 1)
        self.assertTrue(parent["duplicate_identity_reserved"][0])
        self.assertEqual(int(parent["partition"][0]), 3)
        self.assertFalse(parent["train_eligible"][0])
        self.assertEqual(len(fitsio.read(str(self.root / "tables/external_test.fits"), ext=1)), 1)

    def test_discarded_duplicate_ambiguity_reservation_survives(self):
        self.check_discarded_reservation_survives("identity_ambiguity_reserved")

    def test_discarded_duplicate_reference_reservation_survives(self):
        self.check_discarded_reservation_survives("external_any")

    def test_spatial_map_is_fixed_and_has_requested_cell_fractions(self):
        config = {"nside": 8, "seed": 20261028, "train_fraction": .7, "validation_fraction": .15}
        first = stage1.make_partition_map(config)
        np.testing.assert_array_equal(first, stage1.make_partition_map(config))
        np.testing.assert_array_equal(np.bincount(first), [538, 115, 115])


if __name__ == "__main__":
    unittest.main()

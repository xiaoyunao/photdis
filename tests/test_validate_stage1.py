"""Independent final-audit tests, including intentionally corrupted products."""

import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import struct
import unittest
from unittest.mock import patch

import fitsio
import numpy as np

import test_stage1_contract as fixture
from photdis import stage1

spec = importlib.util.spec_from_file_location("validate_stage1", Path(__file__).resolve().parents[1] / "scripts/validate_stage1.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FinalAuditTests(unittest.TestCase):
    def make_run(self, root, size=3):
        _, metadata = fixture.Stage1ContractTests().run_tile(fixture.sweep_rows(size), root)
        run_dir = root / "output"
        refs = root / "refs"
        refs.mkdir()
        np.savez(refs / "holdout_references_v1.npz", member_source_id=np.array([102], dtype="i8"),
                 binary_source_id=np.zeros(0, dtype="i8"))
        config = {"reference_dir": str(refs), "sky": {"abs_b_min": 30, "ebv_max": .03},
                  "photometry": {}, "output_dir": str(run_dir), "coverage_reference_modulus": 1}
        stage1.write_json(run_dir / "scan_manifest.json", {
            "signature": "test", "files": [metadata["source"]], "config": config,
            "code_commit": "synthetic", "all_source_tiles": 1,
            "source_bytes": metadata["source_bytes"], "limited_smoke_run": True,
        })
        with patch.object(stage1, "require_compute"), patch.object(stage1, "run_signature", return_value="test"), contextlib.redirect_stdout(io.StringIO()):
            stage1.finalize(config)
        return run_dir

    def validate(self, run_dir):
        with patch.object(audit, "require_compute"):
            return audit.validate_run(run_dir, chunk_rows=2, sample_size=2)

    def test_actual_pipeline_products_and_empty_products_pass(self):
        for size in (0, 3):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as root:
                run_dir = self.make_run(Path(root), size)
                result = self.validate(run_dir)
                self.assertTrue(result["passed"], result["checks"])
                self.assertEqual(result["counts"]["parent"], size)
                self.assertTrue((run_dir / "validation.json").exists())
                self.assertLessEqual(result["coverage"]["parent"]["sample_rows"], 2)
                self.assertEqual(result["scan_manifest_sha256"],
                                 hashlib.sha256((run_dir / "scan_manifest.json").read_bytes()).hexdigest())
                for role in stage1.ROLE_NAMES:
                    table = fitsio.read(str(run_dir / "tables" / f"{role}.fits"), ext=1,
                                        columns=["ls_id", "source_id", "partition"])
                    canonical = b"".join(struct.pack("<qqh", int(row["ls_id"]), int(row["source_id"]),
                                                    int(row["partition"])) for row in table)
                    self.assertEqual(result["role_identity_partition_sha256"][role],
                                     hashlib.sha256(canonical).hexdigest())
                with patch.object(audit, "require_compute"):
                    rechunked = audit.validate_run(run_dir, chunk_rows=1, sample_size=2)
                self.assertEqual(result["role_identity_partition_sha256"],
                                 rechunked["role_identity_partition_sha256"])
                if size:
                    self.assertEqual(result["counts"]["negative_parallax"], 1)
                    self.assertEqual(result["counts"]["zero_parallax"], 1)
                    self.assertEqual(result["anchor_cutflow_on_retained_measured_parent"]["final_anchor_after_contaminants"], 1)

    def test_lost_negative_parallax_and_duplicate_identity_are_detected(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = self.make_run(Path(root))
            path = run_dir / "tables/parent.fits"
            with fitsio.FITS(str(path), "rw") as hdus:
                extended = hdus[1].read_column("extended")
                extended[0] = False
                hdus[1].write_column("extended", extended)
                ids = hdus[1].read_column("source_id")
                ids[1] = ids[0]
                hdus[1].write_column("source_id", ids)
            result = self.validate(run_dir)
            self.assertFalse(result["passed"])
            self.assertEqual(result["checks"]["extended_preserves_signed_parallaxes"]["violations"], 1)
            self.assertEqual(result["checks"]["unique_gaia_ids"]["violations"], 1)

    def test_missing_role_member_is_detected(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = self.make_run(Path(root))
            path = run_dir / "tables/anchor.fits"
            table = fitsio.read(str(path), ext=1)
            fitsio.write(str(path), table[:0], extname="STAGE1", clobber=True,
                         header={"RUNHASH": "test", "ROLE": "anchor"})
            result = self.validate(run_dir)
            self.assertFalse(result["passed"])
            self.assertEqual(result["checks"]["anchor_exact_identity_set"]["violations"], 1)
            self.assertEqual(result["checks"]["anchor_summary_count"]["violations"], 1)

    def test_shared_spectra_must_be_reserved_together(self):
        checks = audit.Checks()
        summary = audit.check_spectrum_partitions(np.array(["a", "a", "b", "c"]),
                                                  np.array([3, 3, 0, 1]), checks, "desi")
        self.assertTrue(checks.passed)
        self.assertEqual(summary["shared_objects"], 1)
        checks = audit.Checks()
        audit.check_spectrum_partitions(np.array(["a", "a", "b", "b"]),
                                       np.array([0, 2, 1, 1]), checks, "desi")
        self.assertFalse(checks.passed)
        self.assertEqual(checks.results["desi_spectrum_cross_partition"]["violations"], 1)
        self.assertEqual(checks.results["desi_shared_spectrum_reserved"]["violations"], 2)

    def test_real_validator_requires_allocated_compute(self):
        with patch.dict(audit.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "allocated NERSC"):
                audit.require_compute()


if __name__ == "__main__":
    unittest.main()

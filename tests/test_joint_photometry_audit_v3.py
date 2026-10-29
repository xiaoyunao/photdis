import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from joint_photometry_audit_v3 import audit_decision


def test_independent_r_band_audit_requires_multiple_indicators():
    audit = {"gaia_source_id": 42, "sweep_row": {"ref_id": 42, "rchisq_r": 2097},
             "ps1": {"matched_within_1_arcsec": 1},
             "legacy_minus_ps1_decam": {"g": -.011, "r": .463, "z": .022}}
    assert audit_decision(audit) == "r_band_discordant"
    audit["sweep_row"]["rchisq_r"] = 1
    with pytest.raises(ValueError):
        audit_decision(audit)

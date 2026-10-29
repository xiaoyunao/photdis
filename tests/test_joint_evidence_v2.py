import json

import fitsio
import numpy as np
import pytest

from joint_calibration_evidence_v2 import recover_redrock_type


def test_redrock_type_recovers_by_identity_not_row_position(tmp_path):
    field = tmp_path / "field"
    field.mkdir()
    rows = np.array([(8, 80, b"QSO"), (7, 70, b"STAR")],
                    dtype=[("ls_id", "i8"), ("source_id", "i8"),
                           ("desi_spectype", "S8")])
    fitsio.write(str(field / "field_diagnostics.fits"), rows)
    field_config = tmp_path / "field.json"
    field_config.write_text(json.dumps({"name": "f", "output_dir": str(field)}))
    config = tmp_path / "joint.json"
    config.write_text(json.dumps({"fields": [{"config": str(field_config)}]}))
    prior = np.array([(b"f", 7, 70), (b"f", 8, 80)],
                     dtype=[("field", "S8"), ("ls_id", "i8"),
                            ("source_id", "i8")])
    assert recover_redrock_type(prior, config).tolist() == [b"STAR", b"QSO"]
    prior["source_id"][1] = 999
    with pytest.raises(ValueError, match="Gaia identity mismatch"):
        recover_redrock_type(prior, config)

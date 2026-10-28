import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cold_v4_type_boundary import failed_edges


def test_optical_box_failure_names_are_per_boundary():
    assert failed_edges(.3, 1.5, [.4, .2], [1.4, 1.2]) == [
        "gr0_below_low", "rz0_above_high"]
    assert failed_edges(.5, .5, [.4, .2], [1.4, 1.2]) == []

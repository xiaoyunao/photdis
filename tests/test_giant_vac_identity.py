import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_giant_vac_identity import match_one


def test_match_one_distinguishes_presence_and_unique_identity():
    catalog = np.array([5, 3, 5, 8])
    location, unique = match_one(np.array([3, 5, 7]), catalog)
    assert location[0] == 1
    assert location[1] in (0, 2)
    assert location[2] == -1
    assert unique.tolist() == [True, False, False]

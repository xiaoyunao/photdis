import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from giant_vac_inventory import exact_pair_join, giant_mask


def test_exact_pair_join_requires_both_ids_and_unique_pair():
    catalog = np.array([(10, 100), (10, 200), (11, 300), (11, 300)],
                       dtype=[("TARGETID", "i8"), ("SOURCE_ID", "i8")])
    found = exact_pair_join(np.array([10, 10, 11, 12]),
                            np.array([200, 999, 300, 100]), catalog)
    assert found.tolist() == [1, -1, -1, -1]


def test_giant_selection_accepts_fits_unicode_and_bytes():
    for dtype in ("U28", "S28"):
        rows = np.array([("conservative_giant",), ("conservative_dwarf",)],
                        dtype=[("class_evidence", dtype)])
        assert giant_mask(rows).tolist() == [True, False]

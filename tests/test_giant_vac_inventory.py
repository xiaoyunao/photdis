import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from giant_vac_inventory import (deduplicate_giant_identities, exact_pair_join,
                                 giant_mask)


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


def test_deduplication_keeps_source_and_target_keys_on_selected_rows():
    rows = np.array([
        (200, 20, b"1002", 12.0),
        (100, 10, b"1001", 20.0),
        (200, 21, b"1003", 30.0),
        (300, 30, b"1004", 15.0),
    ], dtype=[("source_id", "i8"), ("ls_id", "i8"),
              ("desi_object_id", "S20"), ("desi_snr", "f4")])
    chosen, target, source = deduplicate_giant_identities(rows)
    assert chosen["source_id"].tolist() == [100, 200, 300]
    assert target.tolist() == [1001, 1003, 1004]
    assert source.tolist() == [100, 200, 300]
    catalog = np.array([(1001, 100), (1003, 200), (1004, 300)],
                       dtype=[("TARGETID", "i8"), ("SOURCE_ID", "i8")])
    assert exact_pair_join(target, source, catalog).tolist() == [0, 1, 2]

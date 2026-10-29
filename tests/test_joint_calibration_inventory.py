import numpy as np

from joint_calibration_inventory import select_unique


def test_missing_gaia_ids_do_not_collapse_distinct_desi_targets():
    rows = np.array([
        (1, 0, True, False, b"101", 12., .2),
        (2, 0, True, False, b"102", 14., .3),
        (3, 99, True, False, b"103", 20., .2),
        (4, 99, True, False, b"104", 10., .2),
        (5, 42, True, True, b"105", 99., .1),
    ], dtype=[("ls_id", "i8"), ("source_id", "i8"), ("desi_match", "?"),
              ("desi_match_ambiguous", "?"), ("desi_object_id", "S32"),
              ("desi_snr", "f4"), ("desi_separation_arcsec", "f4")])
    index, report = select_unique(rows)
    assert rows["ls_id"][index].tolist() == [1, 2, 3]
    assert report["matched_unambiguous_rows"] == 4
    assert report["unique_source_rows"] == 3

import numpy as np
from photdis.v3.systematics import calibration_vectors


def test_shared_calibration_is_chunk_and_identity_order_invariant():
    table = dict(field=np.array([0, 0, -1, -1, 102]), cell=np.array([1, 1, 33, 33, 51]),
        transmission=np.full((5, 6), .98), EBV=np.full(5, .01), band_mask=np.ones((5, 6), bool))
    draws = dict(draw_id=np.array([0, 1]), field_ids=np.array([0]), field_dust_offset=np.array([[.002], [-.001]]),
        global_band_zero_point_mag=np.full((2, 6), .003))
    full, dust = calibration_vectors(table, draws, 1)
    order = np.array([4, 0, 3, 1, 2])
    pieces = [calibration_vectors({k: v[ix] for k, v in table.items()}, draws, 1)[0] for ix in (order[:2], order[2:])]
    np.testing.assert_array_equal(np.concatenate(pieces), full[order])
    assert dust[0] == dust[1] == -.001 and dust[2] == dust[3]
    assert dust[2] != dust[4]

import numpy as np

from v3_2_reference_inventory import inventory_join


def test_sdss_v_row_provenance_and_legacy_exclusions():
    table = dict(sdss_id=np.arange(1, 9), gaia_dr3_source_id=np.arange(101, 109, dtype='i8'),
                 release=np.array(['sdss5', 'dr17', 'dr19', 'sdss5', 'sdss5', 'sdss5', 'sdss5', 'sdss5']),
                 apred=np.array(['1.3', 'dr17', '1.3', '1.3', '1.3', '1.3', '1.3', '1.3']),
                 telescope=np.array(['apo25m']*6+['lco25m', 'apo25m']),
                 n_good_visits=np.array([1]*7+[0]), obj=np.array(['new']*4+['legacy']+['new']*3),
                 sdss4_apogee_id=np.array(['']*5+['legacy']+['']*2))
    left, right, counts = inventory_join(table, table, np.array([104]), np.array(['legacy']))
    np.testing.assert_array_equal(left, [0])
    np.testing.assert_array_equal(right, [0])
    assert counts['distance_fresh_eligible'] == counts['stellar_fresh_eligible'] == 1

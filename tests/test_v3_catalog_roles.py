import numpy as np
from v3_0_catalog_export import catalog_role_flags


def test_exact_large_id_roles_and_actual_fit_distinction():
    # Adjacent Legacy identities must not collapse through float conversion.
    base = 2**60
    ids = np.array([base+1, base+2, base+3, base+4], dtype='i8')
    ref = ids[[2, 0, 1]]
    result = catalog_role_flags(ids, ref, [0, 1, 2], ids[:1], ids[1:2], ids[2:3])
    np.testing.assert_array_equal(result['reference_role'], [1, 2, 0, -1])
    np.testing.assert_array_equal(result['sed_training_identity'], [True, False, False, False])
    np.testing.assert_array_equal(result['source_calibration_identity'], [False, True, False, False])
    np.testing.assert_array_equal(result['population_fit_identity'], [False, False, True, False])

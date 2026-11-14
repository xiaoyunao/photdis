import numpy as np
import pytest

from v3_2_tilt_fit import checked_flags


def test_check_contexts_preserve_exact_psf_identity_and_admission():
    identity = 2344520808917792001
    data = dict(row=np.array([1, 33]), identity=np.array([identity, -12]),
                candidate_rows=np.array([33]))
    checks = [dict(row='1', identity=str(identity), TYPE='PSF', admission='broad', passed='True'),
              dict(row='33', identity='-12', TYPE='PSF', admission='broad', passed='True'),
              dict(row='33', identity='-12', TYPE='PSF', admission='candidate', passed='False')]
    assert checked_flags(checks, data) == {'broad': set(), 'candidate': {33}}
    for replacement in (dict(identity='-13'), dict(TYPE='REX'), dict(admission='broad'), dict(passed='')):
        rejected = [*checks[:-1], dict(checks[-1], **replacement)]
        with pytest.raises(ValueError, match='entry check contexts'):
            checked_flags(rejected, data)

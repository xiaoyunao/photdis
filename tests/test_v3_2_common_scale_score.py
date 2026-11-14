import numpy as np
import pytest

from v3_2_common_scale_score import metrics, shared_distances


def test_shared_scale_pairs_cancel_for_identical_forecasts_and_keep_admission():
    q = np.array([[12., 13., 14., 15., 16.], [13., 14., 15., 16., 17.]])
    forecasts = {
        'candidate': dict(mu_quantiles=q, numerical_pass=np.array([True, True])),
        'qrf': dict(mu_quantiles=q, available=np.array([True, True]),
                    numerical_pass=np.array([True, False])),
    }
    result = metrics(forecasts, {'scale_system': np.array([100002, 100002])},
                     np.array([False, True]), {100002: ('example', 7., .4, .6)})
    paired = [r for r in result if r['candidate_minus_view']]
    assert len(paired) == 24
    for row in paired:
        if row['n']:
            assert row['lo95'] == row['median'] == row['hi95'] == 0
        else:
            assert row['registered'] == 1 and row['numerical_only']
            assert row['median'] is None

    x = shared_distances(7., .4, .6, 100002)
    np.testing.assert_array_equal(x, shared_distances(7., .4, .6, 100002))
    assert len(x) == 4096 and (x > 0).all()
    with pytest.raises(ValueError, match='Positive finite'):
        shared_distances(7., 0, .6, 100002)

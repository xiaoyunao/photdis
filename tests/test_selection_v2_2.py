import numpy as np
import pytest
from photdis.selection_v2_2 import canonical_probability, reliability_bins, wilson


def test_endpoints_conserve_all_rows_including_roundoff():
    q = np.array([-1e-16, 0., .1, .5, .95, 1., 1+2e-16])
    out = reliability_bins(q, np.ones(len(q), bool))
    assert sum(r['n'] for r in out) == len(q)
    assert out[0]['n'] == 2 and out[-1]['n'] == 3
    np.testing.assert_array_equal(canonical_probability(q)[1:-1], q[1:-1])
    assert reliability_bins([], [])[0]['n'] == 0


@pytest.mark.parametrize('q', [[np.nan], [np.inf], [-.01], [1.001]])
def test_invalid_probabilities_are_not_silently_clipped(q):
    with pytest.raises(ValueError):
        canonical_probability(q)


def test_wilson_zero_and_full_are_finite_and_asymmetric():
    lo, hi = wilson(0, 100)
    assert abs(lo) < 1e-15 and .03 < hi < .04
    lo, hi = wilson(100, 100)
    assert .96 < lo < .97 and abs(hi-1) < 1e-15

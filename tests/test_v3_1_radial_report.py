import numpy as np
from v3_1_radial_report import quantile


def test_weighted_count_interval_uses_posterior_mass_not_equal_draws():
    values=np.array([8,0,3,9],dtype='i8')
    result=quantile(values,np.array([.02,.01,.96,.01]))
    np.testing.assert_array_equal(result,[3,3,8])
    assert result.dtype==np.int64


def test_weighted_interval_ignores_zero_mass_and_is_permutation_invariant():
    values=np.array([-100.,2.,4.,100.]); weights=np.array([0.,.5,.5,0.])
    np.testing.assert_array_equal(quantile(values,weights),[2.,2.,4.])
    order=np.array([3,1,0,2])
    np.testing.assert_array_equal(quantile(values[order],weights[order]*17),quantile(values,weights))

import numpy as np

from v3_0_catalog_summary import numerical_flags


def test_reference_failure_and_marginal_mc_failure_are_distinct():
    # Opposite response errors cancel in the final mixture. A reference failure
    # still disqualifies its row, while an independent large final error remains.
    a=np.zeros((3,3,6));b=np.zeros_like(a)
    a[0,0,0]=.008;b[0,0,0]=-.008
    a[2,0,0]=b[2,0,0]=.006
    pieces=[dict(half_a=a,half_b=np.zeros_like(a),reference_numerical_converged=np.array([True,False,True])),
            dict(half_a=b,half_b=np.zeros_like(b),reference_numerical_converged=np.ones(3,bool))]
    delta,reference,mc,failed=numerical_flags(pieces,[.5,.5],0.)
    np.testing.assert_allclose(delta,[0.,0.,.006])
    np.testing.assert_array_equal(reference,[False,True,False])
    np.testing.assert_array_equal(mc,[False,False,True])
    np.testing.assert_array_equal(failed,[False,True,True])
    # Omitted-response uncertainty participates in the same frozen threshold.
    _,_,mc,_=numerical_flags(pieces,[.5,.5],.0025)
    assert mc.all()

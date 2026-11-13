import numpy as np
from photdis.v3_1.candidate_selection import selected_gaussian_probability
from photdis.v3_1.candidate_quadrature import bounded_selected_probability


def test_pruned_queries_bracket_same_gaussian_support_quadrature():
    mean = np.array([[3.,4.,5.],[-10.,-11.,-12.],[1.,2.,3.]])
    cov = np.broadcast_to(np.array([[1.,.3,.1],[.3,1.5,-.2],[.1,-.2,2.]]),(3,3,3))
    lower = np.array([1.,1.2,1.]); cal = np.array([.01,-.02,.03])
    support = lambda m:(m[:,0]-m[:,1] > -.5) & (m[:,1]-m[:,2] < 1.)
    old = selected_gaussian_probability(mean,cov,lower,cal,support=support,draws=512,seed=31)
    new,bound = bounded_selected_probability(mean,cov,lower,cal,support=support,draws=512,seed=31)
    assert (old >= new-1e-15).all() and (old <= new+bound+1e-15).all()
    assert (bound <= 1e-14).all() and bound[1] > 0
    exact,zero = bounded_selected_probability(mean,cov,lower,cal,support=support,draws=512,seed=31,support_weight_floor=0.)
    np.testing.assert_allclose(exact,old,atol=1e-15,rtol=0)
    np.testing.assert_array_equal(zero,0.)

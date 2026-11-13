import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.stats import norm
from photdis.v3_1.radial import RadialPopulation


def fixture(local):
    grid = np.linspace(-15,45,3001)
    centers = np.array([[11,14,16],[12,13,15],[15,17,12.]])
    likelihood = norm.pdf(grid[None,None,:],centers[:,:,None],.4)
    entry = np.broadcast_to(norm.sf((grid-18)/2),likelihood.shape)
    return RadialPopulation(grid,likelihood,entry,[.7,.1,.2],local=local)


@pytest.mark.parametrize('local',[False,True])
def test_radial_analytic_gradient(local):
    model = fixture(local); z = np.random.default_rng(3).normal(0,.2,model.dimension)
    error = check_grad(lambda x:model.objective(x)[0],lambda x:model.objective(x)[1],z)
    assert error < 2e-5


def test_selection_cancels_matching_curves():
    model = fixture(True)
    model = RadialPopulation(model.mu,model.entry,model.entry,[.7,.1,.2],local=True)
    z = np.random.default_rng(5).normal(0,.3,19)
    value,gradient = model.objective(z)
    np.testing.assert_allclose(value,.5*z@z,atol=1e-12)
    np.testing.assert_allclose(gradient,z,atol=1e-12)


def test_local_narrow_curve_and_resolution_guard():
    model = fixture(True); grid = model.mu
    local,_ = model.local_terms(13.013,.08)
    expected = norm.pdf(13.013,np.array([[11,14,16],[12,13,15],[15,17,12.]]),np.hypot(.4,.08))
    np.testing.assert_allclose(local[:,:,0],expected,rtol=1e-10,atol=1e-14)
    coarse = RadialPopulation(grid[::25],model.response[:,:,::25],model.entry[:,:,::25],[.7,.1,.2],local=True)
    with pytest.raises(ValueError,match='unresolved'): coarse.local_terms(13.013,.08)

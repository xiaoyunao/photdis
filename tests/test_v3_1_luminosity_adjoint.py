"""Adjoint identity must hold before assigning corrected modes to latent kernels."""
import numpy as np
import pytest
from scipy.special import logsumexp
from photdis.v3_1.luminosity_log import log_luminosity_transform
from photdis.v3_1.luminosity_adjoint import log_luminosity_adjoint


@pytest.mark.parametrize('delta,sigma',[(0.,0.),(.27,0.),(-.27,.18),(.27,.18),(0.,.4)])
def test_log_inner_product_with_boundaries_and_tiny_likelihood(delta,sigma):
    grid=np.linspace(-5,5,101)
    native=-20000-.5*((grid+4.8)/.24)**2
    weights=np.stack([-.5*((grid+4.7)/.16)**2, .6*grid, -.5*((grid-4.9)/.1)**2])
    forward=log_luminosity_transform(native,grid,delta,sigma)
    adjoint=log_luminosity_adjoint(weights,grid,delta,sigma)
    np.testing.assert_allclose(logsumexp(forward+weights,axis=-1),logsumexp(native+adjoint,axis=-1),atol=1e-10,rtol=0)


@pytest.mark.parametrize("delta,sigma",[(.37,.2),(0.,.3),(-.2,.2)])
def test_adjoint_against_explicit_basis_matrix(delta,sigma):
    grid=np.linspace(-3,3,61); basis=np.full((61,61),-np.inf);np.fill_diagonal(basis,0.)
    # Rows are unit native-grid basis functions; columns are true-mu outputs.
    matrix=log_luminosity_transform(basis,grid,delta,sigma)
    logweight=-.5*((grid-.9)/.2)**2
    expected=logsumexp(matrix+logweight[None],axis=1)
    actual=log_luminosity_adjoint(logweight,grid,delta,sigma)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def test_zero_mass_and_identity_are_preserved():
    grid=np.linspace(-3,3,61); missing=np.full((2,len(grid)),-np.inf)
    assert np.isneginf(log_luminosity_adjoint(missing,grid,-.3,.2)).all()
    value=np.linspace(-20000,-19999,len(grid))
    np.testing.assert_array_equal(log_luminosity_adjoint(value,grid,0.,0.),value)


def test_nonuniform_or_nonfinite_weights_rejected():
    with pytest.raises(ValueError,match='Uniform'):
        log_luminosity_adjoint(np.zeros(4),np.array([0.,1.,2.,4.]),0.,.1)
    with pytest.raises(ValueError,match='log weights'):
        log_luminosity_adjoint(np.array([0.,np.nan,0.]),np.arange(3.),0.,.1)

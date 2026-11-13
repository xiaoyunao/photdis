import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from photdis.v3_1.structure import (withdrawn_gaussian_logresponse,selected_structure_loglike,
    shared_member_counts,structure_quadrature)


def test_gaussian_baseline_withdraws_prior_before_applying_new_prior():
    for m,s,a,b in [(14.,.4,14.8,.02),(17.,2.,11.5,1.5),(8.,7.,20.5,3.5)]:
        calculated=withdrawn_gaussian_logresponse(-20000.,m,s,[a],[b])[0]+20000.
        # Independently integrate in standardized new-prior coordinates.
        integral=quad(lambda z:np.exp(norm.logpdf(a+b*z,m,s)-norm.logpdf(a+b*z,15,5))*norm.pdf(z),-12,12,epsabs=1e-12)[0]
        np.testing.assert_allclose(calculated,np.log(integral),atol=2e-9)
    with pytest.raises(ValueError,match='Nonintegrable'):
        withdrawn_gaussian_logresponse(0,10,10,[15],[10])


def test_selection_normalizes_catalog_but_does_not_divide_membership_components():
    ll,p=selected_structure_loglike(np.log([.4]),np.log([.1]),[.2],[.8],[0.],0.)
    np.testing.assert_allclose(ll,np.log(.25/.5))
    np.testing.assert_allclose(p,[.8])


def test_global_uncertainty_yields_correlated_fixed_total_counts():
    p=np.array([[.1,.1],[.9,.9]]);r=shared_member_counts(p,[.5,.5])
    np.testing.assert_allclose(r['mean'],[1,1])
    np.testing.assert_allclose(r['conditional_variance'],.18)
    np.testing.assert_allclose(r['shared_variance'],.64)
    np.testing.assert_allclose(r['covariance'],[[.82,-.82],[-.82,.82]])
    np.testing.assert_allclose(r['covariance'].sum(axis=1),0)


def test_structure_quadrature_represents_one_shared_prior():
    mu,lw,z,lz=structure_quadrature(32,128)
    np.testing.assert_allclose(np.exp(lw).sum(),1)
    np.testing.assert_allclose(np.exp(lz).sum(),1)
    np.testing.assert_allclose(np.exp(lz)@(z*z),16,rtol=1e-12)
    assert (np.diff(mu)>0).all()

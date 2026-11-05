import numpy as np

from photdis.weak_wise import JointWISE


def test_signed_flux_and_missing_bands_on_two_luminosity_branches():
    rng = np.random.default_rng(5)
    grz = np.tile([18.6, 18.2, 18.], (96, 1)) + rng.normal(0, .004, (96, 3))
    absolute_g = np.repeat([1., 6.], 48)
    ir = grz[:, 2, None] - np.repeat([[1.5, 2.], [0., .15]], 48, axis=0)
    model = JointWISE(grz, absolute_g, ir, np.arange(-3, 12.1, .1))
    observed = np.tile([18.6, 18.2, 18.], (5, 1))
    error = np.full_like(observed, .02)
    valid = np.array([[0, 0], [1, 0], [1, 1], [1, 1], [1, 1]], bool)
    flux = np.array([[np.nan, np.nan], [-1., np.nan], [0., -1.], [251., 398.], [251., 398.]])
    ivar = np.array([[np.nan, np.nan], [1., np.nan], [1., 1.], [1., 1.], [1e-12, 1e-12]])
    pdf, ess = model.predict(observed, error, np.arange(5), flux, ivar, np.ones_like(flux), valid)
    np.testing.assert_allclose(pdf.sum(axis=2), 1)
    np.testing.assert_array_equal(pdf[0, 0], pdf[0, 2])
    np.testing.assert_array_equal(pdf[1, 1], pdf[1, 2])
    np.testing.assert_allclose(pdf[4, 0], pdf[4, 2], atol=1e-7)
    # Negative flux favors the faint IR template; high flux favors the bright one.
    assert pdf[1, 1, 70:].sum() > pdf[1, 0, 70:].sum()
    assert pdf[3, 2, :70].sum() > pdf[3, 0, :70].sum()
    assert np.isfinite(ess).all()

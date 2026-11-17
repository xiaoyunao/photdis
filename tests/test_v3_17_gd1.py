"""Regression tests for GD-1 background and external identity claims."""
import numpy as np

from v3_17_hess import detect, weights
from v3_17_members import resolve
from v3_17_sample import NEW


def test_background_rejects_a_linear_gradient_and_recovers_an_injection():
    far, blanks = [1, 2, 3, 4], [5, 6, 7, 8]
    pattern = np.asarray([1., .8, .7, 1., .5, .6, .7, .5, .4])
    w = weights(pattern, NEW, far, 0.)
    np.testing.assert_allclose([w@pattern, w@(pattern*NEW)], [1., 0.], atol=1e-12)
    exposure = np.broadcast_to(pattern, (3,len(pattern)))
    counts = exposure*(100+2*NEW)
    control, _, _ = detect(counts, exposure, NEW, far, blanks, (0,1,2))
    assert not control['signal']
    injected = counts.copy()
    injected[:,0] += 120
    hit, _, _ = detect(injected, exposure, NEW, far, blanks, (0,1,2))
    assert hit['signal']
    np.testing.assert_allclose(hit['excess'], 360)


def test_gaia_identifier_can_disambiguate_but_must_not_conflict():
    exact = ((0,1,123,.10,'PSF',True,True),'a')
    wrong = ((0,2,456,.05,'PSF',False,True),'b')
    position = ((0,2,456,.05,'PSF',False,False),'b')
    assert resolve([wrong,exact]) == (exact, 'gaia_id')
    assert resolve([wrong]) == (None,'gaia_conflict')
    assert resolve([position,wrong]) == (None,'ambiguous')
    assert resolve([position]) == (position,'unique_position')

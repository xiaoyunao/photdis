"""Check the GD-1 footprint wrap and CMD selection function."""
import numpy as np

from v3_16_fit import magnitude_response
from v3_16_gd1 import ROTATION, angles, bounding_box, coordinates, unit
from v3_16_sample import APPARENT, COLOR


def test_footprint_wrap_uses_fits_right_ascension():
    phi1 = np.array([-39., -25., -10.1])
    phi2 = np.array([-.8, .0, .8])
    ra, dec = angles(ROTATION.T @ unit(phi1, phi2))
    ra %= 360
    ra0, dec0, ra1, dec1 = bounding_box()
    assert ra0 < ra.min() < 180 < ra.max() < ra1
    assert dec0 < dec.min() < dec.max() < dec1
    np.testing.assert_allclose(coordinates(ra, dec), [phi1, phi2], atol=1e-7)


def test_cmd_projection_respects_observed_magnitude_tiers():
    centers = .5*(APPARENT[1:]+APPARENT[:-1])
    weight = magnitude_response(np.array([.022, .034]))
    grid = weight.reshape(3, len(COLOR)-1, len(centers))
    assert np.all(grid[0][:, centers>=20.] == 0)
    assert np.all(grid[2][:, centers<20.9] == 0)
    np.testing.assert_allclose(grid.sum(axis=0)[:, centers<21.9], 1)
    np.testing.assert_allclose(grid.sum(axis=0)[:, centers>=22.05], 0)

import numpy as np
import pytest
from photdis.v3_1.radial_selection import RadialSelectionTable


def test_entry_profile_and_error_translation():
    profiles = np.array([[0,0],[1,0],[0,1.]])
    mu = np.linspace(-10,40,1001)
    curves = .6-.005*mu[None,None,:]+.03*profiles[:,0,None,None]+.02*profiles[:,1,None,None]
    curves = np.broadcast_to(curves,(3,3,len(mu))).copy()
    table = RadialSelectionTable(profiles,mu,curves)
    grid = np.linspace(5,20,101); s = 2.
    errors = np.array([[s*np.exp(.2),s,s*np.exp(.3)]])
    value,audit = table.evaluate(errors,grid)
    expected = .6-.005*(grid+2.5*np.log10(s))+.03*.2+.02*.3
    np.testing.assert_allclose(value[0],np.tile(expected,(3,1)),atol=1e-14)
    assert audit['inside_profile_hull'][0]
    _,audit = table.evaluate([[np.exp(2),1,np.exp(2)]],grid)
    assert not audit['inside_profile_hull'][0]
    with pytest.raises(ValueError,match='outside entry table'): table.evaluate(errors,[-10,40])

import numpy as np
from photdis.v3_1.sky import qualified_sky


def test_bad_coordinates_do_not_poison_other_rows():
    ra=np.array([192.85948,0.,np.nan,180.,30.,-1.,360.,12.])
    dec=np.array([27.12825,0.,np.nan,0.,40.,0.,0.,91.])
    result=qualified_sky(ra,dec);valid=np.array([True,True,False,True,True,False,False,False])
    np.testing.assert_array_equal(result['valid_coordinates'],valid)
    clean=qualified_sky(ra[valid],dec[valid])
    np.testing.assert_allclose(result['latitude'][valid],clean['latitude'],atol=1e-12)
    np.testing.assert_allclose(result['latitude'][valid][:3],[90.,-60.18855,60.18855],atol=1e-4)
    assert (result['cell'][~valid]==-1).all() and np.isnan(result['latitude'][~valid]).all()
    assert np.sum(abs(result['latitude'])>30)==3


def test_empty_and_polar_coordinates():
    assert len(qualified_sky([],[])['cell'])==0
    result=qualified_sky([0.,359.9],[90.,-90.])
    np.testing.assert_array_equal(result['cell'],[17,35*18])

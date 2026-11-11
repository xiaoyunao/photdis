import numpy as np
from astropy.io import fits
from v2_4_reserve_fast import native_keys


def test_native_sequential_reader_preserves_ids_and_morphology(tmp_path):
    path=tmp_path/'native.fits'
    columns=[fits.Column(name='RELEASE',format='I',array=[11010,11010,11010]),
             fits.Column(name='BRICKID',format='J',array=[123456,123456,98765]),
             fits.Column(name='OBJID',format='J',array=[0,1234567,2]),
             fits.Column(name='TYPE',format='4A',array=['PSF','EXP','']),
             fits.Column(name='unused',format='16D',array=np.arange(48).reshape(3,16))]
    fits.BinTableHDU.from_columns(columns).writeto(path)
    actual=native_keys(path)
    np.testing.assert_array_equal(actual['RELEASE'],[11010]*3)
    np.testing.assert_array_equal(actual['BRICKID'],[123456,123456,98765])
    np.testing.assert_array_equal(actual['OBJID'],[0,1234567,2])
    np.testing.assert_array_equal(np.char.strip(actual['TYPE'].astype('U')),['PSF','EXP',''])

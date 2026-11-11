import numpy as np
from photdis.v3.data import observations, specdis_qualification, primary_channel, assign_roles


def test_specdis_qualification_has_no_binary_or_gaia_selection():
    good, sigma = specdis_qualification([2, 3, -1, 2], [.2, .3, .1, .2], [1, 1, 1, 0])
    np.testing.assert_array_equal(good, [True, True, False, False])
    np.testing.assert_allclose(sigma[:2], 5/np.log(10)*np.sqrt(np.log1p(.01)))
    np.testing.assert_array_equal(primary_channel(good, [1, 1, 1, 0], [1]*4), [1, 1, 2, 3])


def test_negative_wise_is_observed_not_missing_and_north_i_absent():
    raw = {'TYPE': np.array(['PSF', 'PSF', 'EXP'])}
    for band in ('G', 'R', 'I', 'Z', 'W1', 'W2'):
        for key, value in [('FLUX', [-2., 10, 1]), ('FLUX_IVAR', [1., 1, 1]),
                           ('MW_TRANSMISSION', [.99]*3), ('NOBS', [1]*3)]:
            raw[key+'_'+band] = np.array(value)
    for band in ('W1', 'W2'):
        raw['FRACFLUX_'+band] = np.zeros(3)
        raw['WISEMASK_'+band] = np.array([0, 1, 0])
    obs = observations(raw, ['DECam_DR11', 'BASS_MzLS_DR9', 'DECam_DR11'], [1]*3, [3]*3)
    assert obs['band_state'][0, 4] == 3 and obs['flux'][0, 4] == -2
    assert not obs['band_mask'][1, 4] and not obs['band_mask'][1, 2]
    assert not obs['psf'][2]
    del raw['FLUX_W2']
    obs = observations(raw, 'DECam_DR11', [1]*3, [3]*3)
    assert not obs['band_mask'][:, 5].any() and np.isnan(obs['flux'][:, 5]).all()


def test_roles_identity_and_calibration_sky_isolation():
    split = np.array([0, 1, 1, 2, 3, 1, 2])
    cell = np.array([1, 2, 2, 3, 4, 2, 5])
    role = assign_roles(split, cell, [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1])
    assert role[1] == role[2] and role[1] in [2, 3]
    np.testing.assert_array_equal(role[[0, 3, 4, 5, 6]], [1, 4, 0, 1, 5])

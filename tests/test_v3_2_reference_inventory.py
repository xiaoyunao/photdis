import fitsio
import numpy as np
import pytest

import v3_2_reference_controls as controls
from test_v3_inference import model, observation
from v3_2_reference_controls import external_calibration
from v3_2_reference_evidence import MINESWEEPER_COLS, read_minesweeper
from v3_2_reference_freeze import verify_request_pointers
from v3_2_reference_inventory import inventory_join


def test_sdss_v_row_provenance_and_legacy_exclusions():
    table = dict(sdss_id=np.arange(1, 9), gaia_dr3_source_id=np.arange(101, 109, dtype='i8'),
                 release=np.array(['sdss5', 'dr17', 'dr19', 'sdss5', 'sdss5', 'sdss5', 'sdss5', 'sdss5']),
                 apred=np.array(['1.3', 'dr17', '1.3', '1.3', '1.3', '1.3', '1.3', '1.3']),
                 telescope=np.array(['apo25m']*6+['lco25m', 'apo25m']),
                 n_good_visits=np.array([1]*7+[0]), obj=np.array(['new']*4+['legacy']+['new']*3),
                 sdss4_apogee_id=np.array(['']*5+['legacy']+['']*2))
    left, right, counts = inventory_join(table, table, np.array([104]), np.array(['legacy']))
    np.testing.assert_array_equal(left, [0])
    np.testing.assert_array_equal(right, [0])
    assert counts['distance_fresh_eligible'] == counts['stellar_fresh_eligible'] == 1


def test_minesweeper_fits_projection_preserves_ids_and_original_row_order(tmp_path):
    schema = {'source_id': 'i8', 'obs': 'S3', 'flag': 'i8'}
    data = np.ones(3, dtype=[(name.upper(), schema.get(name, 'f8')) for name in MINESWEEPER_COLS])
    data['SOURCE_ID'] = np.array([2**60 + 11, 2**60 + 12, 2**60 + 13])
    data['OBS'] = b'LCO'
    data['FLAG'] = [0, 0, 1]
    data['DIST'] = [4., 7., 15.]
    data['DIST_LERR'] = [.1, .1, .1]
    data['DIST_UERR'] = [.3, 1.5, .5]
    data['TEFF'] = 5000.
    data['LOGG'] = [2., 5., 2.]
    data['LOGG_LERR'] = .2
    data['LOGG_UERR'] = .2
    filename = tmp_path/'source.fits'
    fitsio.write(str(filename), data)
    rows = [2, 1, 0]
    index = dict(reference_row=np.array(rows), identity=data['SOURCE_ID'][rows], obs=np.full(3, 'LCO'))
    evidence = read_minesweeper(filename, index)
    np.testing.assert_allclose(evidence['distance_kpc'], [np.nan, 7., 4.])
    np.testing.assert_array_equal(evidence['distance_precise'], [False, False, True])
    np.testing.assert_array_equal(evidence['class_label'], [-1, 0, 1])
    with pytest.raises(AssertionError):
        read_minesweeper(filename, dict(index, identity=index['identity'] + 1))


def test_field_freeze_rechecks_pointers_against_closed_requests():
    ids = np.array([2**60 + 7, 2**60 + 11, 2**60 + 9])
    requests = dict(identity=ids, reference_row=np.array([9, 17, 12]), obs=np.full(3, 'LCO'))
    index = {key: values[[1, 0]] for key, values in requests.items()}
    verify_request_pointers(index, requests, ('reference_row', 'obs'))
    with pytest.raises(ValueError, match='pointer differs'):
        verify_request_pointers(dict(index, reference_row=index['reference_row'][::-1]),
                                requests, ('reference_row', 'obs'))
    with pytest.raises(ValueError, match='identity absent'):
        verify_request_pointers(dict(index, identity=ids[[0, 2]] + 1), requests, ('reference_row',))


def test_external_calibration_uses_sky_cells_without_native_field_ids():
    photo = dict(identity=np.arange(3), cell=np.array([4, 2, 4]),
                 transmission=np.full((3, 6), .9), EBV=np.full(3, .02),
                 band_mask=np.ones((3, 6), bool))
    draws = dict(draw_id=np.array([12]), field_ids=np.array([4, 2]),
                 field_dust_offset=np.array([[.1, .2]]),
                 global_band_zero_point_mag=np.zeros((1, 6)))
    cal = external_calibration(photo, draws)
    np.testing.assert_array_equal(cal[0], cal[2])
    assert not np.allclose(cal[0], cal[1])
    assert np.abs(cal).max() < .1
    assert 'field' not in photo
def test_reference_controls_predict_both_views_on_uniform_grids(monkeypatch):
    sed = model()
    obs = dict(observation(), EBV=0.)
    photo = {key: np.asarray([value]) for key, value in obs.items()}
    grids = controls.forecast_grids()
    theta = {0: np.full((2, 3, 5), 1/15)}
    monkeypatch.setattr(controls, '_STATE', (
        photo, sed, sed, np.zeros((1, 6)), grids, [0], theta, np.zeros((2, 2)), np.full(2, .5),
    ))
    views = controls.continuous_views(0)
    assert set(views) == {'nominal_reference', 'old_target'}
    for rows in views.values():
        for grid, (logpdf, quantiles, probabilities, _) in zip(grids, rows, strict=True):
            assert logpdf.shape == grid.shape and np.isfinite(quantiles).all()
            assert probabilities.shape == (3, 6)
            np.testing.assert_allclose(probabilities.sum(), 1., atol=1e-12)

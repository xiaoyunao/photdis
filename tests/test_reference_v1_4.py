import numpy as np
from photdis.reference_v1_4 import spatial_split, unique_join, pair_keys, optional_photometry, optical_features
from photdis.photometry import BANDS


def test_partial_legacy_overlap_quarantines_whole_cell():
    cell, split = spatial_split([1, 9, 11, np.nan], [1, 9, 1, 0], [[8, 8, 9, 9]])
    assert split[0] == split[1] == split[3] == 3
    assert split[2] in (0, 1, 2)


def test_split_is_order_independent_and_cell_level():
    ra = np.array([1, 2, 41, 42, 131, 132])
    dec = np.array([1, 8, -5, -2, 31, 38])
    c, s = spatial_split(ra, dec, [])
    assert np.array_equal(s, spatial_split(ra[::-1], dec[::-1], [])[1][::-1])
    assert np.all(s[::2] == s[1::2])
    assert np.all(c[::2] == c[1::2])


def test_duplicate_exact_pairs_are_rejected_not_arbitrarily_selected():
    cat = pair_keys([8, 9, 8, 8], [2, 3, 2, 4])
    q = pair_keys([8, 8, 9, 7], [2, 4, 3, 8])
    assert unique_join(q, cat).tolist() == [-1, 3, 1, -1]
    assert unique_join(q, cat[:0]).tolist() == [-1] * 4


def test_weak_negative_wise_flux_is_measurement_not_magnitude():
    raw = {'MASKBITS': np.zeros(6, dtype='i8')}
    for b in ['I', 'W1', 'W2']:
        raw.update({f'FLUX_{b}': np.array([10, -1, 10, 10, 10, 10.]),
                    f'FLUX_IVAR_{b}': np.ones(6), f'MW_TRANSMISSION_{b}': np.ones(6),
                    f'NOBS_{b}': np.array([1, 1, 0, 1, 1, 1]), f'FRACFLUX_{b}': np.zeros(6)})
    raw.update(FRACMASKED_I=np.zeros(6), FRACIN_I=np.ones(6),
               WISEMASK_W1=np.array([0, 0, 0, 1, 0, 0]), WISEMASK_W2=np.zeros(6))
    raw['FLUX_IVAR_W1'][4] = 0
    raw['MASKBITS'][5] = 1 << 14
    out = optional_photometry(raw)
    assert out['w1_measurement_state'].tolist() == [4, 3, 0, 2, 1, 4]
    assert np.isnan(out['i0'][1]) and not out['i_usable'][5]
    assert out['i_usable'][0]


def test_explicit_features_preserve_legacy_band_order():
    m = np.array([[20., 19., 18.5, 18.]])
    assert optical_features(m, 'grz').tolist() == [[1., 1.]]
    assert optical_features(m, 'griz').tolist() == [[1., .5, .5]]
    assert BANDS == ('G', 'R', 'Z', 'W1', 'W2')


def test_sweep_extraction_keeps_parent_with_recovered_gaia_id(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import fitsio
    import baseline_v1_4_extract as ex
    from photdis.stage1 import legacy_ids
    types = {'REF_CAT':'S2', 'TYPE':'S3'}
    integer = {'RELEASE', 'BRICKID', 'OBJID', 'REF_ID', 'MASKBITS', 'FITBITS', 'WISEMASK_W1', 'WISEMASK_W2'}
    raw = np.zeros(3, dtype=[(k, types.get(k, 'i8' if k in integer else 'f8')) for k in ex.RAW_COLUMNS])
    raw['RELEASE'], raw['BRICKID'], raw['OBJID'] = 11010, 123, [1, 2, 3]
    raw['REF_ID'] = [111, 0, 333]
    raw['REF_CAT'] = [b'G3', b'', b'G3']
    raw['TYPE'] = b'PSF'
    for b in ['G', 'R', 'I', 'Z', 'W1', 'W2']:
        for p, v in [('FLUX', 20), ('FLUX_IVAR', 1), ('NOBS', 1), ('MW_TRANSMISSION', 1)]:
            raw[f'{p}_{b}'] = v
    for b in ['G', 'R', 'I', 'Z']:
        raw[f'FRACIN_{b}'] = 1
    raw['FLUX_W1'][1] = -2
    path = tmp_path / 'sweep-000p000-005p005.fits'
    fitsio.write(str(path), raw)
    (tmp_path / 'photometry').mkdir()
    monkeypatch.setattr(ex, 'OUT', tmp_path)
    monkeypatch.setattr(ex, 'SOURCE_IDS', np.array([111]), raising=False)
    monkeypatch.setattr(ex, 'PARENT_IDS', np.array([222]), raising=False)
    monkeypatch.setattr(ex, 'PARENT_LSIDS', legacy_ids(raw)[1:2], raising=False)
    result = ex.scan(path)
    assert result['scanned_rows'] == 3 and result['selected_rows'] == 2
    with np.load(tmp_path / 'photometry' / (path.stem + '.npz')) as saved:
        assert saved['OBJID'].tolist() == [1, 2]
        assert saved['FLUX_W1'][1] == -2
        assert saved['w1_measurement_state'][1] == 3
    assert ex.scan(path) == result

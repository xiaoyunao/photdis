"""Exercise the actual small FITS extraction/serialization path with fake fields."""
import json
from pathlib import Path
import sys

import fitsio
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import giant_information_extract as extract


@pytest.mark.parametrize('matched', [True, False])
def test_small_fits_extraction_preserves_missing_feh_class_and_empty_labels(tmp_path, monkeypatch, matched):
    cfg = json.loads(Path('configs/giant_information_v1.json').read_text())
    registry = tmp_path/'registry.json'
    cfg.update(output_dir=str(tmp_path/'output'), registries=[str(registry)])
    path = tmp_path/'experiment.json'; path.write_text(json.dumps(cfg))
    boolean = {'grz_base', 'grz_clean', 'w1_clean', 'w2_clean', 'desi_match', 'desi_match_ambiguous'}
    dtype = [(name, '?' if name in boolean else 'i8' if name in ('ls_id', 'source_id', 'desi_quality_reason')
              else 'S16' if name in ('desi_object_id', 'desi_spectype') else 'f8') for name in extract.COLUMNS]
    entries = []
    for field in range(14):
        directory = tmp_path/f'field{field}'; directory.mkdir()
        rows = np.zeros(4, dtype=dtype)
        rows['ls_id'] = (np.int64(11010)<<40) + (field<<20) + np.arange(1, 5)
        rows['source_id'] = field*10 + np.arange(1, 5)
        # Same Gaia identity in two fields, with agreeing conservative labels.
        if field==1:
            rows['source_id'][0] = 1
        rows['ra'] = 180.; rows['dec'] = 0.; rows['ebv'] = .01
        rows['r'] = rows['r0'] = 17.; rows['g0'] = 18.; rows['z0'] = 16.5
        rows['w10'] = rows['w20'] = 16.
        for name in extract.ERRORS:
            rows[name] = .03
        for name in ('grz_base', 'grz_clean', 'w1_clean', 'w2_clean'):
            rows[name] = True
        rows['snr_w1'] = rows['snr_w2'] = 30.
        rows['desi_match'] = matched
        rows['desi_object_id'] = [str(field*10+i) for i in range(1, 5)]
        rows['desi_spectype'] = [b'STAR', b'STAR', b'STAR', b'QSO']
        rows['desi_snr'] = 50.
        rows['desi_teff'] = 4500.; rows['desi_teff_err'] = 40.
        rows['desi_logg'] = [2.5, 4.8, 3.8, 2.5]; rows['desi_logg_err'] = .1
        rows['desi_feh'] = -1.; rows['desi_feh_err'] = .1
        rows['desi_feh'][0] = np.nan; rows['desi_quality_reason'][0] = 1<<4
        fitsio.write(str(directory/'field_diagnostics.fits'), rows)
        spec = tmp_path/f'field{field}.json'
        spec.write_text(json.dumps({'name': f'field{field}', 'output_dir': str(directory)}))
        entries.append({'config': str(spec), 'role': 'development'})
    registry.write_text(json.dumps({'fields': entries}))
    monkeypatch.setattr(extract, 'CONFIG', path)
    # Only tiny synthetic files are read locally; production guard stays intact.
    monkeypatch.setattr(extract, 'require_compute', lambda: None)
    extract.main()
    result = json.loads((tmp_path/'output/inventory.json').read_text())
    assert result['n_clean_photometric_records'] == 56
    assert result['n_unique_clean_ls_measurements'] == 56
    with np.load(tmp_path/'output/labels.npz', allow_pickle=False) as t:
        if matched:
            assert result['n_spectral_identity_groups'] == 55
            assert result['n_crossfield_groups'] == 1
            assert result['post_identity_K_labels'] == {'giants': 13, 'dwarfs': 14}
            giants = t['label']==1
            assert np.all(~t['abundance_calibration_valid'][giants])
            assert (t['label']==-1).sum() == 28
        else:
            assert result['n_spectral_identity_groups'] == 0
            assert result['n_conflicting_groups'] == 0
            assert len(t['label']) == 0

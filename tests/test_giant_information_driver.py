"""Small synthetic integration tests; no survey data or remote compute used."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from giant_information_fit import fit_field, basic_selection, panel_masks, transfer_summary
from giant_information_extract import availability, add_counts
from giant_information_summarize import figures, report


def synthetic_table():
    rng = np.random.default_rng(112)
    n = 180
    field = np.repeat(np.arange(3), 60)
    y = np.tile(np.r_[np.zeros(30, int), np.ones(30, int)], 3)
    g = 17 + rng.normal(0, .15, n)
    gr = .8+.2*y+rng.normal(0, .03, n)
    rz = .4+.05*y+rng.normal(0, .03, n)
    z = g-gr-rz
    t = {'field_index': field, 'label': y, 'release': np.full(n, 11010),
         'identity_group': np.arange(n), 'ls_id': np.arange(n)+1,
         'representative': np.ones(n, bool), 'identity_conflict': np.zeros(n, bool),
         'desi_teff': rng.uniform(4200, 4999, n), 'desi_logg': np.where(y, 2.5, 4.8),
         'abundance_calibration_valid': np.ones(n, bool),
         'desi_feh_err': np.full(n, .1), 'feh_sigma': np.full(n, np.sqrt(.02)),
         'feh_sigma_random': np.full(n, .1),
         'feh_corrected': -1+2*(gr-1)+rng.normal(0, .15, n),
         'g0': g, 'r0': g-gr, 'r': g-gr+.03, 'z0': z,
         'w10': z-.5-.2*y, 'w20': z-.45-.25*y,
         'snr_w1': np.full(n, 30.), 'snr_w2': np.full(n, 30.),
         'w1_clean': np.ones(n, bool), 'w2_clean': np.ones(n, bool)}
    for b in ('g', 'r', 'z', 'w1', 'w2'):
        t['sigma_'+b] = np.full(n, .025)
    return t


def test_full_fold_paired_predictions_and_insufficient_camera(tmp_path):
    cfg = json.loads(Path('configs/giant_information_v1.json').read_text())
    t = synthetic_table()
    inventory = {'fields': [{'field': f'field_{i}'} for i in range(3)]}
    rec = fit_field(t, inventory, cfg, 0, 'DECam', 'common5', panel_masks(t, cfg)['common5'], tmp_path)
    assert rec['status'] == rec['abundance_status'] == 'scored'
    assert rec['training_fields'] == [1, 2]
    assert set(rec['models']) == {'A', 'B', 'C'}
    assert set(rec['abundance']) == {'marginal', 'A', 'B', 'C'}
    scores = np.load(tmp_path/'DECam_0_common5_scores.npz')
    assert len(scores['score_A']) == len(scores['score_C']) == 60
    assert len(scores['feh_logp_A']) == len(scores['feh_logp_C']) == 30
    assert np.all(t['field_index'][scores['test_rows']]==0)
    assert np.all(t['label'][scores['abundance_test_rows']]==1)
    # FeH quality cannot change class membership.
    before = basic_selection(t, cfg).copy()
    t['abundance_calibration_valid'][:] = False
    np.testing.assert_array_equal(basic_selection(t, cfg), before)
    t['release'][t['field_index']==0] = 11011
    north = fit_field(t, inventory, cfg, 0, 'BASS_MzLS', 'optical', np.ones(180, bool), tmp_path)
    assert north['status'].startswith('insufficient')
    summary = transfer_summary([rec], cfg)
    assert not summary['DECam']['common5']['abundance']['A']['stable_gain']


def test_availability_does_not_turn_missing_wise_into_optical_loss():
    cfg = json.loads(Path('configs/giant_information_v1.json').read_text())
    t = synthetic_table()
    t['w2_clean'][:90] = False
    stats = availability(t, np.ones(180, bool), cfg)
    assert stats['total']['optical'] == stats['total']['w1_snr5'] == 180
    assert stats['total']['w1w2_snr5'] == 90
    result = {}; add_counts(result, stats); add_counts(result, stats)
    assert result['total']['optical'] == 360
    assert sum(result['histograms']['r']['w1w2_snr5']) == 180


def test_empty_camera_summary_is_explicit_not_a_fake_result(tmp_path):
    cfg = json.loads(Path('configs/giant_information_v1.json').read_text())
    inventory = {'fields': [], 'n_clean_photometric_records': 0,
                 'n_unique_clean_ls_measurements': 0, 'n_spectral_records': 0,
                 'n_spectral_identity_groups': 0, 'n_conflicting_groups': 0,
                 'n_crossfield_groups': 0}
    result = {'config': cfg, 'inventory': inventory, 'folds': [],
              'transfer': transfer_summary([], cfg)}
    result['figures'] = figures(result, tmp_path/'figures')
    assert len(result['figures']) == 4
    assert all(Path(p).stat().st_size>1000 for p in result['figures'])
    text = report(result)
    assert 'no stable gain established' in text
    assert 'passes registered limited-family' not in text

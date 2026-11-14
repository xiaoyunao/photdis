import numpy as np
import pytest

from v3_2_reference_freeze import VIEWS, read_admission
from v3_2_reference_score import class_confusion, reliability, score_one, summarize


def test_frozen_admission_checks_identity_order_and_count(tmp_path):
    photo = dict(identity=np.array([101, 102, 103]), ls_id=np.array([201, 202, 203]),
                 TYPE=np.full(3, 'PSF'))
    path = tmp_path/'candidate_admission.npz'
    np.savez(path, **photo, candidate=[True, False, True])

    np.testing.assert_array_equal(read_admission(path, photo, 2), [True, False, True])
    with pytest.raises(ValueError, match='admission mask'):
        read_admission(path, photo, 3)
    with pytest.raises(AssertionError, match='ls_id'):
        read_admission(path, dict(photo, ls_id=photo['ls_id'][::-1]), 2)
    np.savez(path, **photo, candidate=[1, 0, 1])
    with pytest.raises(ValueError, match='admission mask'):
        read_admission(path, photo, 2)


def test_metrics_and_pairs_use_candidate_admission_within_reference_groups():
    evidence = dict(distance_kpc=np.array([3., 4., 6., 4.]), cell=np.array([0, 1, 2, 3]),
                    class_label=np.array([1, 1, 1, 0]), distance_precise=np.ones(4, bool),
                    feh_valid=np.ones(4, bool), feh=np.full(4, -2.))
    rows = {name: [dict(teacher_log_score=float(i), teacher_score_stable=True,
                        numerical_pass=i != 3) for i in range(4)] for name in VIEWS}
    rows['candidate'][1]['teacher_log_score'] = 3.
    absolute, paired = summarize(rows, evidence, np.array([True, False, True, True]))

    def qualified(result, **keys):
        return {(r['admission_scope'], r['numerical_only']): (r['n'], r['mean'])
                for r in result if all(r[k] == value for k, value in keys.items())}

    assert qualified(absolute, view='candidate', group='all_2_5_teacher_kpc',
                     metric='teacher_log_score') == {
        ('native_photometric', False): (3, 2.),
        ('native_photometric', True): (2, 1.5),
        ('matched_candidate_admission', False): (2, 1.5),
        ('matched_candidate_admission', True): (1, 0.),
    }
    assert qualified(paired, anchor='qrf', group='metal_poor_giant_2_5_teacher_kpc',
                     metric='teacher_log_score') == {
        ('native_photometric', False): (2, 1.),
        ('native_photometric', True): (2, 1.),
        ('matched_candidate_admission', False): (1, 0.),
        ('matched_candidate_admission', True): (1, 0.),
    }


def test_gaia_pit_tails_use_signed_observations_and_keep_missing_channels():
    data = dict(numerical_pass=[True], mu_quantiles=np.array([[9.1, 9.3, 10., 10.7, 10.9]]),
                mu_edges=np.array([[9., 10., 11.]]), mass=np.array([[.5, .5]]))
    evidence = dict(parallax_eligible=True, distance_available=False, distance_precise=False,
                    p_corr=-.5, gaia_parallax_error=.1)
    low = score_one('qrf', data, 0, evidence)
    assert low['gaia_pit_low'] and not low['gaia_pit_high']
    high = score_one('qrf', data, 0, dict(evidence, p_corr=2.))
    assert high['gaia_pit_high'] and not high['gaia_pit_low']
    missing = score_one('qrf', data, 0, dict(evidence, parallax_eligible=False))
    assert 'gaia_pit_low' not in missing and 'gaia_pit_high' not in missing


def test_probability_bins_and_counts_keep_identical_qualified_denominators():
    evidence = dict(class_label=np.array([1, 0, -1, 1]), cell=np.arange(4),
                    distance_precise=np.array([True, True, True, False]),
                    distance_kpc=np.array([3., 10., 4., np.nan]))
    forecasts = [dict(qnear=q, p_giant=p, p_dwarf=1-p, numerical_pass=i != 1,
                      teacher_score_stable=True, class_score_stable=True)
                 for i, (q, p) in enumerate(zip([1., .9, .2, .6], [.8, .1, .3, .7]))]
    rows = {name: forecasts for name in VIEWS}
    rows['qrf'] = [{k: v for k, v in row.items() if k not in ('p_giant', 'p_dwarf')}
                   for row in forecasts]
    bins, totals = reliability(rows, evidence, np.array([True, True, False, True]))

    for total in totals:
        matching = [row for row in bins if all(
            row[key] == total[key] for key in ('view', 'target', 'admission_scope', 'numerical_only'))]
        assert sum(row['n'] for row in matching) == total['n']
        assert sum(row['n']*(row['mean'] or 0) for row in matching) == pytest.approx(total['observed_count'])
        assert sum(row['n']*(row['mean_probability'] or 0) for row in matching) == pytest.approx(
            total['expected_count'])

    native_near, matched_near = [
        next(row for row in totals if row['view'] == 'candidate' and row['target'] == 'near'
             and row['admission_scope'] == scope and not row['numerical_only'])
        for scope in ('native_photometric', 'matched_candidate_admission')]
    assert (native_near['reference_qualified'], native_near['observed_count']) == (3, 2)
    assert (matched_near['reference_qualified'], matched_near['observed_count']) == (2, 1)
    assert matched_near['count_residual'] == pytest.approx(.9)
    qrf_giant = next(row for row in totals if row['view'] == 'qrf' and row['target'] == 'giant')
    assert qrf_giant['reference_qualified'] == 3 and qrf_giant['n'] == 0


def test_absolute_confusion_conserves_reference_counts_and_numerical_exclusions():
    evidence = dict(class_label=np.array([1, 1, 0, -1, 1]), cell=np.arange(5))
    forecasts = [dict(predicted_class=p, numerical_pass=i != 1, class_score_stable=True)
                 for i, p in enumerate([0, 2, 0, 1, 1])]
    rows = {name: forecasts for name in VIEWS}
    rows['qrf'] = [dict(numerical_pass=True) for _ in forecasts]
    reports = class_confusion(rows, evidence, np.array([True, True, True, True, False]))
    for row in reports:
        assert row['classified']+row['excluded_numerical']+row['missing_forecast'] == row['reference_qualified']
    subset = [row for row in reports if row['view'] == 'candidate'
              and row['reference_class'] == 'giant'
              and row['admission_scope'] == 'matched_candidate_admission']
    assert [row['count'] for row in subset] == [1, 0, 1, 1, 0, 0]
    assert [row['excluded_numerical'] for row in subset] == [0, 0, 0, 1, 1, 1]
    assert all(row['classified'] == 0 and row['mean'] is None
               for row in reports if row['view'] == 'qrf')

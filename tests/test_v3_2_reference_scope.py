import numpy as np
import pytest

from v3_2_reference_freeze import VIEWS, read_admission
from v3_2_reference_score import summarize


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

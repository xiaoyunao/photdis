import numpy as np
import pytest
from photdis.v3_1.comparison import match_controls, paired_mean


def test_paired_weighted_changes_do_not_treat_repeated_cells_as_independent():
    x = np.array([-2.,1.,4.]); w = np.array([1.,2.,1.])
    r = paired_mean(x,w,[8,8,9])
    assert r['mean'] == 1. and r['cells'] == 2 and r['effective_n'] == 16/6
    assert r['lo95'] <= r['mean'] <= r['hi95']
    single = paired_mean(x,w,[8,8,8])
    assert single['lo95'] is None and single['hi95'] is None


def test_pairing_is_identity_based_and_requires_baseline():
    template = dict(TYPE='PSF',cell=4,stratum='x',inclusion_probability=.5,class_label=1)
    records = [dict(template,identity=2,control='class'),dict(template,identity=1,control='response12'),
               dict(template,identity=2,control='response12')]
    pairs = match_controls(records,'class')
    assert len(pairs) == 1 and pairs[0][0]['identity'] == pairs[0][1]['identity'] == 2
    with pytest.raises(ValueError): match_controls(records[:2],'class')
    with pytest.raises(ValueError): match_controls(records+records[:1],'class')

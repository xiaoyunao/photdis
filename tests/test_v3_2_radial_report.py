import pytest

from v3_2_radial_report import count_loss_comparison, coverage


def test_count_loss_uses_same_catalog_and_all_joint_bins():
    design = dict(design='fixed', source='model', background='near',
                  fraction='0.1', width='0.08', n='128')
    designs = [dict(design, catalog=str(i), repeat=str(i)) for i in range(3)]
    rows = []
    for i, arms in enumerate([('broad', 'local'), ('broad', 'local'), ('broad',)]):
        for arm in arms:
            for j in range(18):
                rows.append(dict(catalog=i, arm=arm, metric=f'count_{j//6}_{j%6}',
                                 truth=128 if j == 0 else 0,
                                 median=(124 if arm == 'broad' else 126) if j == 0 else 0,
                                 interval_length=4 if arm == 'broad' else 3,
                                 accepted=i == 0 or arm == 'broad'))
    paired, summary = count_loss_comparison(designs, rows)
    assert [row['catalog'] for row in paired] == [0, 1]
    assert paired[0]['count_l1_change'] == pytest.approx(-2/128)
    assert paired[0]['mean_interval_change'] == -1
    assert [(row['n'], row['registered']) for row in summary] == [(2, 3), (2, 3), (1, 3), (1, 3)]

    with pytest.raises(ValueError, match='Missing joint count bins'):
        count_loss_comparison(designs, rows[1:])


def test_coverage_bounds_retain_missing_registered_catalogs():
    fitted = [dict(median='4', truth='4', covered95='True', interval_length='2'),
              dict(median='6', truth='4', covered95='False', interval_length='2')]
    result = coverage(fitted, 8)
    assert result['coverage'] == .5
    assert result['coverage_lower_all'] == .125
    assert result['coverage_upper_all'] == .875
    assert result['coverage_lo95'] < .5 < result['coverage_hi95']
    assert coverage([], 8)['coverage_upper_all'] == 1

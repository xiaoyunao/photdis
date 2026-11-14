import pytest

from v3_2_radial_report import (ROOT, count_loss_comparison, coverage, read_csv,
                               shrinkage_summary, validate_interval, validate_numerical)


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


def test_empirical_pilot_recovery_verdicts_and_intervals():
    receipt = ROOT/'manifests/v3_2/radial_recovery/catalog_128'
    broad, local = read_csv(receipt/'numerical.csv')
    assert validate_numerical(broad) and validate_numerical(local)
    absolute_only = dict(local, integration_objective_shift_max='0.1',
                         absolute_objective_pass='False')
    assert validate_numerical(absolute_only)
    with pytest.raises(ValueError, match='differs'):
        validate_numerical(dict(local, fixed_draw_count_pass='False'))
    assert not validate_numerical(dict(status='no_converged_map', sampler_pass='False'))
    rows = read_csv(receipt/'recovery.csv')
    for row in rows:
        validate_interval(row)
    width = next(row for row in rows if row['metric'] == 'mu_width')
    assert width['covered95'] == 'False'
    with pytest.raises(ValueError, match='Invalid recovery interval'):
        validate_interval(dict(width, covered95='True'))


def test_prior_shrinkage_preserves_pilot_denominators():
    root = ROOT/'manifests/v3_2'
    designs = read_csv(root/'radial_design/design.csv')
    rows = [dict(r, catalog=128, accepted=True) for r in
            read_csv(root/'radial_recovery/catalog_128/recovery.csv')]
    results, summary = shrinkage_summary(designs, rows)
    observed = {r['metric']: r for r in results}
    assert observed['mu_width']['prior_anchor'] == pytest.approx(.25)
    assert observed['mu_width']['toward_anchor'] == pytest.approx(.08126487275)
    assert observed['mu_location']['toward_anchor'] < 0
    covered = [r for r in summary if r['n']]
    assert len(covered) == 6
    assert all(r['n'] == 1 and r['registered'] == 8 for r in covered)
    assert all(r['mean_toward_anchor'] is None for r in summary if not r['n'])

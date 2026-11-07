import numpy as np
from photdis.report_v2 import percent_metrics, ranges


def test_percentage_metrics_are_computed_per_star_and_keep_tails():
    ref=np.array([10.,10.,10.,np.nan])
    delta=5*np.log10(np.array([.5,1,2,1]))
    mu=np.repeat((ref+delta)[:,None],5,axis=1)
    mu[:,0]-=2;mu[:,1]-=1;mu[:,3]+=1;mu[:,4]+=2
    result=percent_metrics(mu,ref)
    assert result['n']==3
    np.testing.assert_allclose(result['median_pct'],0)
    np.testing.assert_allclose(result['nmad_pct'],1.4826*50)
    np.testing.assert_allclose(result['tail1mag'],2/3)
    assert ranges([1,np.nan,3])['missing']==1
    assert ranges([1,10],[99,1])['p50']<2


def test_revision_source_lookup_never_wraps_missing_row(tmp_path):
    from v2_0_anomalies import selected_columns
    path = tmp_path / 'source.npz'
    np.savez(path, exact_id=np.array([11, 9223372036854775806], dtype='i8'),
             parameter=np.array([4., 9.]), text=np.array(['first', 'last']), flags=np.array([1,2],dtype='u2'))
    result = selected_columns(path, np.array([-1, 0, 1]), ['exact_id', 'parameter', 'text', 'flags'])
    assert result['exact_id'].tolist() == [-1, 11, 9223372036854775806]
    assert np.isnan(result['parameter'][0])
    assert result['text'][0] == ''
    assert result['flags'].tolist() == [0,1,2]


def test_distance_display_clip_is_simultaneous_and_handles_empty_constant():
    from v2_0_revision import distance_clip
    x = np.r_[np.ones(100), 100, 1, 100, np.nan]
    y = np.r_[np.full(100, 2.), 2, 200, 200, 2]
    keep, stats = distance_clip(x, y)
    assert keep.sum() == 100
    assert not keep[100:].any()
    assert stats['x_rejected'] == stats['y_rejected'] == 2
    assert stats['both_rejected'] == 1 and stats['clipped_n'] == 3
    assert stats['invalid_n'] == 1
    np.testing.assert_allclose(stats['y_mean_kpc'], np.mean(y[:103]))
    assert distance_clip([1, 1], [2, 2])[0].all()
    assert not distance_clip([np.nan, -1], [2, 2])[0].any()
    assert distance_clip([], [])[1]['retained_n'] == 0

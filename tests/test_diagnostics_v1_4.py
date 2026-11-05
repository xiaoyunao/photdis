import numpy as np
from photdis.diagnostics_v1_4 import (distance_kpc,residual_stats,paired_stats,
                                     cell_area,fixed_examples,wise_joint)


def test_distance_units_and_tail_sign():
    assert np.allclose(distance_kpc([10,15,20]),[1,10,100])
    q=np.array([[9,10,11,12,13],[4,5,6,7,8.]])
    r=residual_stats(q,np.array([11,11]))
    assert r['tail_count']==1 and r['bias_mag']==-2.5
    assert r['coverage68']==.5 and r['coverage95']==.5
    assert r['min_delta']==-5


def test_pair_cluster_ratio_and_transition_conservation():
    # Two unequal sky cells: bootstrap must aggregate stars, not average cell rates.
    r=paired_stats([2,2,0,0],[0,0,2,0],[1,1,1,2])
    assert (r['rescued'],r['harmed'],r['both_bad'],r['both_good'])==(2,1,0,1)
    assert r['tail_difference']==-.25
    assert np.allclose(r['tail_ci95'],[-1/3,0])
    assert r['n']==sum(r[k] for k in ['rescued','harmed','both_bad','both_good'])
    assert paired_stats([],[],[])=={'n':0}


def test_sky_area_and_fixed_selection():
    assert abs(cell_area(np.arange(648)).sum()-4*np.pi*(180/np.pi)**2)<1e-8
    assert cell_area(9)>cell_area(17)
    ids=np.arange(20);m={'a':ids<10,'b':ids<20}
    a=fixed_examples(ids,m);b=fixed_examples(ids,m)
    assert a==b and len(set(a['a']+a['b']))==4


def test_wise_overlapping_failures_remain_visible():
    j=wise_joint([1,1,10,10],[1]*4,[1]*4,[1]*4,[1,0,1,0],[2,0,0,0],[1,2,3,4])
    assert j['low_snr_total_fraction']==.3
    assert j['low_snr_and_quality_fraction']==.1
    assert j['quality_failure_fraction']==.4
    assert j['low_snr_given_quality']==.25
    assert sum(c['n'] for c in j['cells'])==4
    assert np.isclose(sum(c['fraction'] for c in j['cells']),1)

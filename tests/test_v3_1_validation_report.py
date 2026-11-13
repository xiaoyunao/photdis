import numpy as np
import pytest
from photdis.v3_1.validation_report import (common_distance_draws,common_scale_statistics,
    field_statistic,paired_field_statistic,external_strata,metric_arrays)
from photdis.v3.data import WEAK,SIGNIFICANT,MISSING


def test_shared_scale_interval_does_not_shrink_with_duplicate_members():
    d=common_distance_draws(16.775,.872,draws=512)
    q=np.array([[14.,15.,16.,17.,18.],[13.,14.,15.,16.,17.]])
    one=common_scale_statistics(q,d)
    many=common_scale_statistics(np.repeat(q,100,axis=0),d)
    for key in one:np.testing.assert_allclose(one[key],many[key],rtol=0,atol=1e-13)
    np.testing.assert_allclose(one['median_mu_residual'],15.5-(10+5*np.log10(d)),atol=1e-14)
    assert np.std(one['median_mu_residual'])>.09
    shifted=common_scale_statistics(q,d*1.03)
    np.testing.assert_allclose(shifted['median_mu_residual']-one['median_mu_residual'],-5*np.log10(1.03),atol=1e-14)


def test_field_interval_resamples_whole_sky_cells_and_retains_infinite_scores():
    value=np.repeat([0.,1.],100); cells=np.repeat([8,9],100)
    result=field_statistic(value,cells)
    assert result['mean']==.5 and result['lo95']==0 and result['hi95']==1
    single=field_statistic(value,np.zeros(len(value)))
    assert single['lo95'] is None and single['hi95'] is None
    pair=paired_field_statistic([-np.inf,2.,np.nan],[-np.inf,1.,0.],[1,2,3])
    assert pair['n']==2 and pair['nonfinite']==1 and pair['mean'] is None
    assert pair['undefined_nonfinite_pairs']==1


def test_strata_keep_missing_evidence_and_negative_W1_as_weak():
    photo=dict(identity=np.arange(3,dtype='i8'),TYPE=np.array(['PSF']*3),r0=np.array([17.,19.,23.]),
        band_state=np.full((3,6),MISSING),flux=np.ones((3,6)),flux_error=np.ones((3,6)))
    photo['band_state'][:,4]=[WEAK,SIGNIFICANT,MISSING]; photo['flux'][:,4]=[-2,10,np.nan]
    evidence=dict(identity=photo['identity'],TYPE=photo['TYPE'],class_label=np.array([-1,0,1]),
        feh_valid=np.array([False,True,True]),feh=np.array([np.nan,-1.5,-.5]),
        distance_available=np.array([False,True,True]),distance_kpc=np.array([np.nan,3,14]))
    masks=external_strata(photo,evidence)
    np.testing.assert_array_equal(masks['W1/weak'],[True,False,False])
    np.testing.assert_array_equal(masks['distance/unavailable'],[True,False,False])
    assert not masks['class_feh/giant/poor'].any()
    for prefix in ['class/','feh/','distance/','r0/','W1/']:
        np.testing.assert_array_equal(np.array([m for name,m in masks.items() if name.startswith(prefix)]).sum(axis=0),1)


def test_empty_shared_subset_is_explicit_and_invalid_intervals_fail():
    result=common_scale_statistics(np.empty((0,5)),np.array([10.,11.]))
    assert all(np.isnan(value).all() for value in result.values())
    with pytest.raises(ValueError,match='Ordered'):
        common_scale_statistics(np.array([[1,2,4,3,5]]),np.array([10.]))


def test_unknown_class_is_missing_and_zero_probability_is_retained():
    base=dict(mu=[1,2,3,4,5],width68=2,width95=4,qnear=.5,references={})
    first=dict(views={'v':base},checks=[dict(view='v',accepted=True)])
    second=dict(views={'v':dict(**base,**{'class':dict(label=0,predicted=2,probability=0.,log_score=None,
        zero_probability=True,brier=2.)})},checks=[dict(view='v',accepted=False)])
    values,accepted=metric_arrays([first,second],'v')
    assert np.isnan(values['class/log_score'][0]) and np.isneginf(values['class/log_score'][1])
    np.testing.assert_array_equal(accepted,[True,False])

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
    evidence.update(specdis_valid=np.array([True,False,False]),specdis_distance=np.array([14.,np.nan,np.nan]),
        mws_class_label=np.array([1,-1,-1]),mws_parameter_quality=np.array([True,False,False]),
        rvs_feh=np.array([-1.5,np.nan,np.nan]),rvs_feh_err=np.array([.1,np.nan,np.nan]))
    masks=external_strata(photo,evidence)
    assert masks['distance/unavailable'][0] and masks['specdis_distance/10_20'][0]
    assert masks['class/unknown'][0] and masks['mws_class_feh/giant/poor'][0]


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


def test_common_report_keeps_primary_domain_and_same_draw_paired_contrasts(tmp_path,monkeypatch):
    import csv
    import json
    import v3_1_validation_report as driver
    from photdis.v3_1.validation_report import MAIN_VIEW
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    root=tmp_path/'data/v3_1'; source=root/'final_validation/common'; source.mkdir(parents=True)
    (source/'scales.json').write_text(json.dumps([dict(system='NGC_1261',method='hst_kinematic',
        reference=dict(distance_kpc=16.775,minus_kpc=.872,plus_kpc=.872))]))
    out=tmp_path/'report'; out.mkdir(); baseline='old_target/old_target'; views=[MAIN_VIEW,baseline]
    records=[]
    for i in range(2):
        main=dict(mu=[14,15,16,17,18],width68=2,width95=4,qnear=.98,references={})
        other=dict(main,mu=[14.2,15.2,16.2,17.2,18.2])
        records.append(dict(views={MAIN_VIEW:main,baseline:other},
            checks=[dict(view=MAIN_VIEW,accepted=True),dict(view=baseline,accepted=i==0)]))
    metrics={view:metric_arrays(records,view) for view in views}
    outputs=driver.common_report(root,dict(candidate_admission=np.array([True,False])),
        dict(scale_system=np.array([0,0])),records,views,metrics,out,{})
    assert all((out/name).exists() for name in outputs)
    with (out/'common_scale_paired.csv').open() as stream:
        pairs=list(csv.DictReader(stream))
    selected=[r for r in pairs if r['metric']=='median_mu_residual' and r['subset']=='candidate_members']
    assert len(selected)==6 and all(int(r['n'])==1 for r in selected)
    for row in selected:
        for key in ('lower95','median','upper95'):assert float(row[key])==pytest.approx(-.2,abs=1e-13)
    with (out/'common_scale_summary.csv').open() as stream:
        summary=list(csv.DictReader(stream))
    main=[r for r in summary if r['view']==MAIN_VIEW and r['admission']=='all_objects' and r['method_factor']=='1.0'
          and r['metric']=='median_mu_residual']
    by_subset={r['subset']:r for r in main}
    assert int(by_subset['candidate_members']['n'])==1 and int(by_subset['all_members_diagnostic']['n'])==2
    assert float(by_subset['candidate_members']['lower95'])==float(by_subset['all_members_diagnostic']['lower95'])

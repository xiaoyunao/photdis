import numpy as np
from photdis.v3_1.domain_assessment import assess_pair,summarize_cell


def test_sparse_or_single_sky_never_passes_reference_screen():
    for n,sky in [(19,np.arange(19)%2),(30,np.zeros(30))]:
        result=assess_pair(np.zeros(n),np.zeros(n),sky,np.ones(n,bool),np.ones(n,bool),'log_score')
        assert result['mean_within_margin'] and not result['sufficient']
        assert result['assessment']=='insufficient_references_or_sky'
        assert summarize_cell([result])['relative_assessment']=='unvalidated_sparse_or_missing_reference'


def test_mean_margin_and_correlated_uncertainty_are_distinct():
    n=40; sky=np.repeat([1,2],20); delta=np.repeat([-.04,.02],20)
    result=assess_pair(delta,np.zeros(n),sky,np.ones(n,bool),np.ones(n,bool),'log_score')
    assert result['assessment']=='mean_guard_passed'
    assert result['sky_interval_assessment']=='crosses_margin'
    assert not result['absolute_calibration_established']
    failure=assess_pair(np.full(n,.03),np.zeros(n),sky,np.ones(n,bool),np.ones(n,bool),'tail')
    assert failure['assessment']=='mean_guard_failed' and failure['sky_interval_assessment']=='beyond_margin'
    assert summarize_cell([failure])['relative_assessment']=='relative_reference_guard_failure'


def test_nonfinite_score_and_bad_numerical_pairs_remain_in_denominator():
    n=40; sky=np.arange(n)%4; good=np.ones(n,bool); bad=good.copy();bad[0]=False
    result=assess_pair(np.zeros(n),np.zeros(n),sky,good,bad,'reference_agreement68')
    assert result['n']==n and result['paired_numerical_failed']==1
    assert result['assessment']=='numerical_or_nonfinite_failure_retained'
    a=np.zeros(n);a[0]=-np.inf
    result=assess_pair(a,np.zeros(n),sky,good,good,'log_score')
    assert result['n']==n and result['mean_within_margin'] is None and result['nonfinite']==1
    a[0]=np.nan
    result=assess_pair(a,np.zeros(n),sky,good,good,'log_score')
    assert result['n']==n and result['qualified_nan_pairs']==1 and result['nonfinite']==1


def test_field_cells_keep_independent_class_channels_and_failed_pair_denominators():
    from v3_1_domain_assessment import field_cell_screens
    from photdis.v3_1.domain_assessment import BASELINE
    from photdis.v3_1.validation_report import MAIN_VIEW
    import pytest
    n=40; ids=np.arange(n,dtype='i8')+2**54; admit=np.ones(n,bool);admit[0]=False
    photo=dict(identity=ids,TYPE=np.full(n,'PSF'),candidate_admission=admit,cell=np.arange(n)%2,
        role=np.full(n,'new_sky_field'))
    domain=dict(identity=ids,TYPE=photo['TYPE'],candidate=admit,cell=np.full(n,4))
    evidence=dict(class_label=np.full(n,-1),mws_class_label=np.ones(n),distance_kpc=np.full(n,np.nan),
        distance_available=np.zeros(n,bool),distance_precise=np.zeros(n,bool),parallax_eligible=np.zeros(n,bool),
        specdis_distance=np.full(n,12.),specdis_valid=np.ones(n,bool),specdis_sigma_mu=np.full(n,.1),
        spdist_distance=np.full(n,np.nan),spdist_valid=np.zeros(n,bool),spdist_sigma_mu=np.full(n,np.nan))
    values={'qnear':np.full(n,.98),'specdis/log_score':np.zeros(n),'specdis/severe':np.zeros(n),
        'specdis/reference_agreement68':np.ones(n),'specdis/reference_agreement95':np.ones(n),'mws_class/log_score':np.zeros(n)}
    good=np.ones(n,bool);bad=good.copy();bad[1]=False
    rows,summaries=field_cell_screens(photo,evidence,{MAIN_VIEW:(values,bad),BASELINE:(values,good)},domain)
    select=lambda group,source:next(r for r in rows if r['role']=='new_sky_field' and r['cell']==4 and
        r['group']==group and r['source']==source and r['metric']=='log_score' and r['qualification']=='qualified')
    assert select('mws_class_1','specdis')['n']==39
    assert select('apogee_class_-1','specdis')['n']==39
    assert select('all','astronn')['n']==0
    assert select('all','specdis')['paired_numerical_failed']==1
    cell=next(r for r in summaries if r['role']=='new_sky_field' and r['cell']==4 and r['group']=='all')
    assert cell['n']==39 and cell['relative_assessment']=='numerical_failures_retained'
    incomplete=dict(values);del incomplete['mws_class/log_score']
    with pytest.raises(ValueError,match='complete score channels'):
        field_cell_screens(photo,evidence,{MAIN_VIEW:(incomplete,bad),BASELINE:(values,good)},domain)

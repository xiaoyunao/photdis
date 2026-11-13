import copy
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from photdis.v3_1.validation_scores import score_pdf,compare_score_levels,finite_pdf


def inputs():
    grid=np.linspace(5,25,4001); lp=norm.logpdf(grid,15,.7)
    quantiles=norm.ppf([.025,.16,.5,.84,.975],15,.7)
    probability=np.full((3,6),1/18)
    evidence=dict(distance_available=True,distance_kpc=10.,distance_error_kpc=1.,distance_precise=True,
        parallax_eligible=True,p_corr=-.1,gaia_parallax_error=.2,class_label=-1)
    return grid,lp,quantiles,probability,evidence


def test_external_gaussian_score_and_signed_gaia_keep_unknown_class():
    g,lp,q,p,e=inputs(); r=score_pdf(g,lp,q,p,e)
    expected=norm.logpdf(15,15,np.hypot(.7,5/np.log(10)*.1))
    assert r['references']['astronn']['log_score']==pytest.approx(expected,abs=1e-5)
    assert r['gaia']['negative'] and 0<r['gaia']['pit']<1
    assert 'class' not in r and r['qnear']==pytest.approx(.5)
    assert r['references']['astronn']['reference_agreement68']


def test_common_scale_single_object_integral_matches_independent_quadrature():
    g,lp,q,p,e=inputs(); e['distance_available']=False; e['parallax_eligible']=False
    value=score_pdf(g,lp,q,p,e,common_scale=(16.775,.872))
    expected=quad(lambda mu:norm.pdf(mu,15,.7)*norm.pdf(16.775,10**((mu-10)/5),.872),5,25,epsabs=1e-13)[0]
    assert value['references']['common_shared_scale']['log_score']==pytest.approx(np.log(expected),abs=1e-9)
    assert value['references']['common_shared_scale']['shared']


def test_same_probability_can_fail_proper_score_tail_guard():
    a=score_pdf(*inputs()); b=copy.deepcopy(a)
    b['references']['astronn']['log_score']+=.03
    comparison=compare_score_levels(a,b)
    assert not comparison['passed'] and comparison['channels_checked']==2
    b=copy.deepcopy(a); b['gaia']['pit']+=.004
    assert not compare_score_levels(a,b)['passed']
    b['references'].clear()
    with pytest.raises(ValueError,match='same external channels'):compare_score_levels(a,b)


def test_zero_endpoints_are_trimmed_but_interior_zero_holes_rejected():
    g=np.arange(7.); lp=np.array([-np.inf,-2.,-1.,0.,-1.,-2.,-np.inf])
    actual,_=finite_pdf(g,lp); np.testing.assert_array_equal(actual,g[1:-1])
    lp[3]=-np.inf
    with pytest.raises(ValueError,match='endpoint'):finite_pdf(g,lp)


def test_nine_view_driver_reads_original_reference_and_keeps_view_flags(tmp_path,monkeypatch):
    import json
    import v3_1_validation_score as driver
    g,lp,q,p,e=inputs(); identity=9007199254740993
    evidence={k:np.array([v]) for k,v in e.items()}
    evidence.update(identity=np.array([identity],dtype='i8'),TYPE=np.array(['PSF']),
        role=np.array(['new_sky_field']),cell=np.array([123]),scale_system=np.array([-1]))
    monkeypatch.setattr(driver,'ROOT',tmp_path); monkeypatch.setattr(driver,'_COHORT','field')
    monkeypatch.setattr(driver,'_EVIDENCE',evidence); monkeypatch.setattr(driver,'_COMMON',{})
    monkeypatch.setattr(driver,'_INPUTS',{}); monkeypatch.setattr(driver,'_OUT',tmp_path/'output')
    driver._OUT.mkdir()
    value=dict(mu=q.tolist(),class_distance=p.tolist())
    for family,folder,names in [('reference','validation_reference_refinement',driver.REFERENCE_VIEWS),
            ('old_target','validation_old_target',['old_target']),
            ('candidate','validation_target/candidate',driver.TARGET_VIEWS)]:
        obj=tmp_path/f'data/v3_1/{folder}/field/object_{identity}'; obj.mkdir(parents=True)
        attempts=[]
        for step in ([.01] if family=='reference' else [.02,.01]):
            if family=='old_target':
                meta=dict(**value,passed=True); arrays=dict(logpdf=lp)
            else:
                meta=dict(results={n:dict(**value,inherited_ensemble_accepted=n!='point') for n in names},
                    checks=[{('view' if family=='candidate' else 'variant'):n,'passed':True} for n in names])
                arrays={n+'_logpdf':lp for n in names}
            np.savez(obj/f'step{step:g}.npz',grid=g,**arrays)
            (obj/f'step{step:g}.json').write_text(json.dumps(meta))
            attempts.append(dict(step=step,marker=f'step{step:g}.json'))
        (obj/'complete.json').write_text(json.dumps(dict(attempts=attempts)))
        if family=='reference':
            old=tmp_path/f'data/v3_1/validation_reference_predictions/field'; old.mkdir(parents=True)
            np.savez(old/f'object_{identity}.npz',grid=g,**arrays)
            (old/f'object_{identity}.json').write_text(json.dumps(meta))
    result=driver.one(0)
    assert len(result['views'])==9 and result['identity']==identity
    flags={r['view']:r['accepted'] for r in result['checks']}
    assert not flags['candidate/point'] and flags['candidate/zero']
    assert flags['reference/nominal'] and flags['old_target/old_target']

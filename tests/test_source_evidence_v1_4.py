import numpy as np
from photdis.source_evidence_v1_4 import class_evidence, gaia_input, vac_evidence


def test_raw_class_adapter_does_not_require_iron_distance_or_photometry():
    source = {f'rvs_{k}': np.asarray(v) for k, v in dict(
        sn_r=[20]*4, vsini=[1]*4, rr_spectype=['STAR']*4,
        rvs_warn=[0]*4, success=[True]*4, primary=[True, True, True, False],
        teff=[5000]*4, teff_err=[50]*4, logg=[2, 4.5, 3.8, 2],
        logg_err=[.1]*4, feh=[np.nan]*4, feh_err=[np.nan]*4).items()}
    evidence = class_evidence(source)
    assert evidence['class_label'].tolist() == [1, 0, 2, -1]
    assert evidence['rvs_parameter_quality'].tolist() == [True, True, True, False]


def test_gaia_adapter_decodes_false_and_preserves_negative_parallax():
    source = dict(source_id=np.array([7, 0]), gaia_parallax=np.array([-.2, np.nan]))
    for name in ['duplicated_source', 'in_qso_candidates', 'in_galaxy_candidates']:
        source['gaia_' + name] = np.array(['false', 'unmatched'])
    ga = gaia_input(source)
    assert ga['gaia_match'].tolist() == [True, False]
    assert not ga['duplicated_source'].any() and not ga['in_qso_candidates'].any()
    assert ga['parallax'][0] == -.2


def test_specdis_requires_unique_positive_pair_and_frozen_quality():
    cat = {k: np.asarray(v) for k, v in dict(targetid=[1,1,2,3,4,5], source_id=[11,11,12,0,14,15],
           dist=[1,1,2,2,-1,3], disterr=[.1,.1,.2,.1,.2,.2], binary_flag=[1,1,1,1,1,0]).items()}
    e = vac_evidence(np.arange(1,6), np.array([11,12,0,14,15]), cat, 'specdis')
    assert e['specdis_match'].tolist() == [False,True,False,True,True]
    assert e['specdis_valid'].tolist() == [False,True,False,False,False]
    assert e['specdis_distance'][1] == 2
    assert np.isclose(e['specdis_sigma_mu'][1],5/np.log(10)*np.sqrt(np.log1p(.1**2)))


def test_spdist_quality_is_independent_and_uses_published_quantiles():
    cat = {k:np.asarray(v) for k,v in dict(targetid=[1,2,3,4], source_id=[11,12,13,14],
        flag_good=[True,True,True,False], qflag_input=[True]*4, g0=[18,9,18,18],
        mg_50=[3]*4, mg_16=[2.5]*4, mg_84=[3.5,3.5,2,3.5], e_gmag=[.1]*4).items()}
    e=vac_evidence(np.arange(1,5),np.arange(11,15),cat,'spdist')
    assert e['spdist_valid'].tolist()==[True,False,False,False]
    assert e['spdist_distance'][0]==10
    assert np.isclose(e['spdist_sigma_mu'][0],np.sqrt(.1**2+.5**2+.167**2))

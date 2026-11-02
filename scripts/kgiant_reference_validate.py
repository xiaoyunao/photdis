#!/usr/bin/env python3
"""Cross-artifact, identity and probability validation of the actual batch."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
from scipy.special import ndtr
from astropy.coordinates import SkyCoord
import astropy.units as u
from photdis.stage1 import require_compute,write_json
from kgiant_distance_inventory import sha
from kgiant_reference_batch import OUT,REF

def main():
    require_compute();ref=json.load(open(REF));inv=json.load(open(OUT/'inventory.json'));p=json.load(open(OUT/'predictions.json'));tr=json.load(open(OUT/'training.json'))
    checks={};rr={r['name']:r for r in ref['rows']}
    assert len(rr)==162 and len(ref['membership_mapping'])==170 and len(inv['systems'])==170
    checks['fixed_parent_completeness']=True
    for r in rr.values():
        d=r['methods']['combined'];assert d['distance_kpc']>d['minus_kpc']>0 and d['plus_kpc']>0
        assert np.isclose(r['mu'],10+5*np.log10(d['distance_kpc']))
    checks['reference_units_and_asymmetric_errors']=True
    mapping={r['member_name']:r['reference_name'] for r in ref['membership_mapping']}
    assert all(r['reference_name']==mapping[r['member_system']] for r in inv['rows'])
    checks['system_reference_identity_join']=True
    eligible={r['source_id']:r for r in inv['rows'] if r['eligible']}
    assert len(eligible)==sum(r['eligible'] for r in inv['rows'])
    assert {r['source_id'] for r in p['rows']}==set(eligible)
    checks['frozen_selection_fully_scored']=True
    for r in eligible.values():
        assert r['pmember']>.9 and r['selected_measurement']['release']==11010 and r['selected_measurement']['grz_quality']
        b=SkyCoord(r['ra']*u.deg,r['dec']*u.deg).galactic.b.deg
        assert abs(b-r['galactic_b'])<1e-5 and r['old_latitude_scope']==(abs(b)>30)
        assert r['old_dust_scope']==(0<=r['selected_measurement']['ebv']<.03)
    checks['quality_and_latitude_dust_flags']=True
    refs=np.load(OUT/'all_reference_member_ids.npz');allids=np.r_[refs['source_id'],refs['ambiguous_ids']]
    for t in tr:
        ids=np.array(tr[t]['source_ids'],dtype=np.int64);assert len(ids)==tr[t]['n'] and not np.isin(ids,allids).any()
    checks['all_system_training_identity_isolation']=True
    expected=sum(2*(3 if r['selected_measurement']['w1_quality'] else 2) for r in eligible.values())
    assert len(p['rows'])==expected
    for sid,r in eligible.items():
        for t in tr:
            models={x['model'] for x in p['rows'] if x['source_id']==sid and x['teacher']==t}
            assert models==({'prior','A_grz','B_grzW1'} if r['selected_measurement']['w1_quality'] else {'prior','A_grz'})
    checks['A_not_W1_gated_and_B_pairing']=True
    for r in p['rows']:
        n=OUT/(r['source_id']+'_'+r['teacher']+'_'+r['model']+'.npz')
        with np.load(n) as d:
            assert np.array_equal(d['train_source_id'],np.array(tr[r['teacher']]['source_ids'],dtype=np.int64))
            assert np.all(d['sigmas']>0) and np.all(d['weights']>=0) and np.isclose(d['weights'].sum(),1)
            q=np.array(r['quantiles_mu']);cdf=ndtr((q[:,None]-d['means'])/d['sigmas'])@d['weights']
            assert np.allclose(cdf,[.025,.16,.5,.84,.975],atol=1e-6)
            assert np.isclose(r['effective_n'],1/np.sum(d['weights']**2))
        assert np.allclose(r['distance_kpc_quantiles'],10**((q-10)/5))
        assert np.isclose(r['delta_mu_median'],q[2]-rr[r['reference_name']]['mu'])
    checks['full_mixture_quantiles_weights_and_distance_units']=True
    assert len(p['paired'])==sum(r['model']=='B_grzW1' for r in p['rows'])
    checks['paired_AB_counts']=True
    for path,digest in p['source_sha256'].items():assert sha(path)==digest,path
    assert inv['typed_candidates_sha256']==sha(OUT/'typed_candidates.json')
    checks['frozen_source_hashes']=True
    old=json.load(open('manifests/kgiant_external_predictions_v1.json'))
    historical=[r for r in p['rows'] if r['historical_diagnostic']]
    assert len(historical)==18
    for r in historical:
        oldrow=next(x for x in old['rows'] if x['name']==r['name'] and x['teacher']==r['teacher'] and x['model']==r['model'])
        assert np.allclose(r['quantiles_mu'],oldrow['quantiles_mu'],atol=1e-10,rtol=0)
    checks['all18_historical_predictions_unchanged']=True
    out=dict(checks=checks,passed=len(checks),n_stars=p['n_stars'],n_systems=p['n_systems'],n_predictions=len(p['rows']),
        source_sha256={str(path):sha(path) for path in [OUT/'inventory.json',OUT/'predictions.json',OUT/'training.json',Path(__file__)]})
    write_json(OUT/'validation.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()

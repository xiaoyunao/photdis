#!/usr/bin/env python3
"""Independent real-artifact checks for the fixed external experiment."""
from pathlib import Path
import sys,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
from scipy.integrate import trapezoid
from photdis.stage1 import require_compute,write_json
from photdis.giant_kernel import mixture_quantiles
from kgiant_distance_inventory import sha

def main():
    require_compute();out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/kgiant_external_v1')
    inv=json.load(open(out/'inventory.json'));p=json.load(open(out/'predictions.json'));tr=json.load(open(out/'training.json'))
    checks=[]
    def check(name,ok):
        assert ok,name
        checks.append(name)
    check('fixed_six_identities',[(r['cluster'],r['name']) for r in inv['rows']]==[
        ('Pal12','6817950249538781952')]+[('Terzan7',x) for x in ('1272','1282','1515','1665','1708')])
    check('distinct_quality_scope_flags',all(all(k in r for k in ['grz_quality','w1_quality','old_dust_scope']) for r in inv['rows'] if 'source_id' in r))
    check('Pal12_optical_not_blocked_by_W1',inv['rows'][0]['grz_quality'] and not inv['rows'][0]['w1_quality'])
    z=zipfile.ZipFile('data/reference/vasiliev_baumgardt_2021_members.zip');memberids=set()
    for f in ('Pal_12','Terzan_7'):
        memberids.update(l.split()[0] for l in z.read('catalogues/'+f+'.txt').decode().splitlines() if not l.startswith('#'))
    check('no_external_system_member_in_either_teacher',all(not memberids.intersection(v['source_ids']) for v in tr.values()))
    check('frozen_training_counts',tr['spdist']['n']==77 and tr['specdis']['n']==258)
    check('no_residual_based_eligibility',all(a['eligible']==(not a['reasons']) for a in p['audit']))
    expected=0
    for r,a in zip(inv['rows'],p['audit']):
        if a['eligible']:expected+=2*(2+int(r['w1_quality']))
    check('optical_and_paired_model_completeness',len(p['rows'])==expected)
    check('cold_control_not_scored',not any(r['name']=='1665' for r in p['rows']))
    for r in p['rows']:
        stem=r['cluster']+'_'+r['name']+'_'+r['teacher']+'_'+r['model']
        m=np.load(out/(stem+'.npz'))
        assert np.isclose(m['weights'].sum(),1) and (m['sigmas']>0).all()
        assert np.allclose(mixture_quantiles(m['means'],m['sigmas'],m['weights']),r['quantiles_mu'])
        assert [str(i) for i in m['train_source_id']]==tr[r['teacher']]['source_ids']
        assert np.isclose(r['delta_mu_median'],r['quantiles_mu'][2]-r['reference_mu'])
        assert np.allclose(np.array(r['distance_kpc_quantiles']),10**((np.array(r['quantiles_mu'])-10)/5))
        assert r['distribution_cdf'][0]<1e-5 and r['distribution_cdf'][-1]>1-1e-5
        assert np.all(np.diff(r['distribution_cdf'])>=-1e-12)
        assert abs(trapezoid(r['distribution_pdf'],r['distribution_grid_mu'])-1)<.001
    checks.append('all_scored_distributions_normalized_quantiles_ids_and_distance_units_reproduced' if p['rows'] else 'zero_predictions_consistent_with_independent_failures')
    for path,digest in p['source_sha256'].items():assert sha(Path(path))==digest,path
    checks.append('prediction_source_hashes_match')
    write_json(out/'validation.json',dict(passed=checks,n_checks=len(checks),n_predictions=len(p['rows']),
        hashes={str(f):sha(f) for f in (out/'inventory.json',out/'predictions.json',out/'training.json')}))
    print('PASS',checks,flush=True)

if __name__=='__main__':main()

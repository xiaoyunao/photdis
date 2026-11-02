#!/usr/bin/env python3
"""Check executed reference, model and complete-catalog artifacts in allocation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
import fitsio
from photdis.stage1 import require_compute
from baseline_v0_fit import load,OUT,CFG
from kgiant_distance_inventory import write_json,sha


def main():
    require_compute();t=load();prep=json.load(open(OUT/'prepared.json'));fit=json.load(open(OUT/'fit_result.json'))
    checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:raise AssertionError(name)
    check('unique_reference_identities',len(np.unique(t['identity']))==len(t['identity']))
    check('training_precision',np.all(t['optical_error'][t['fit_eligible']]<.03))
    check('training_has_no_identity_conflict',not t['identity_conflict'][t['fit_eligible']].any())
    check('training_has_no_known_nonstellar',not t['nonstellar_risk'][t['fit_eligible']].any())
    check('A_trained_on_optical_only_objects',fit['models']['A']['n_train_without_W1']>0)
    for a,b in [('A_on_B','B'),('A_on_C','C'),('B_on_C','C')]:
        check(a+'_same_train',fit['models'][a]['train_identity_sha256']==fit['models'][b]['train_identity_sha256'])
    for name,entry in fit['models'].items():
        check(name+'_model_hash',sha(OUT/(name+'.joblib'))==entry['model_sha256'])
        with np.load(OUT/(name+'_evaluation.npz')) as p:
            check(name+'_pdf_normalized',np.allclose(p['pdf'].sum(axis=1),1,atol=2e-5))
            check(name+'_pdf_nonnegative',np.all(p['pdf']>=0))
            check(name+'_ordered_finite_quantiles',np.isfinite(p['mu']).all() and np.all(np.diff(p['mu'],axis=1)>=0))
            check(name+'_no_training_prediction_in_test',np.all(t['split'][p['rows']]>0))
            check(name+'_calibration_contains_raw',np.all(p['mu_cal'][:,:2]<=p['mu'][:,:2]) and np.all(p['mu_cal'][:,3:]>=p['mu'][:,3:]))
    total=0
    for field in range(14):
        name=f'field_{field:02}'
        with np.load(OUT/f'parent_{field:02}.npz') as p:
            expected=p['identity'][np.load(OUT/f'owner_{field:02}.npy')]
        meta=json.load(open(OUT/f'catalog_{field:02}.json'));path=OUT/f'catalog_{field:02}.fits'
        with fitsio.FITS(str(path)) as f:
            n=f[1].get_nrows();check(name+'_row_count',n==len(expected)==meta['n_unique']);total+=n
            for start in range(0,n,100000):
                rows=f[1].read(rows=np.arange(start,min(start+100000,n)))
                check(name+'_identity_order',np.array_equal(rows['identity'],expected[start:start+len(rows)]))
                check(name+'_internal_flag',rows['internal_uncalibrated'].all())
                if field in CFG['north_fields']:
                    check(name+'_no_wrong_system_model',np.all(rows['route']==-1) and np.isnan(rows['mu']).all())
                    continue
                check(name+'_finite_recommendation',np.isfinite(rows['mu']).all() and np.all(rows['route']>=0))
                check(name+'_distance_units',np.allclose(rows['distance_kpc'],10**((rows['mu'].astype(float)-10)/5),rtol=3e-6))
                for k in 'ABC':
                    use=np.ones(len(rows),bool) if k=='A' else rows[k]
                    check(name+'_'+k+'_availability',np.isfinite(rows[k+'_mu'][use]).all() and np.isnan(rows[k+'_mu'][~use]).all())
                    check(name+'_'+k+'_ordered_quantiles',np.all(np.diff(rows[k+'_mu'][use],axis=1)>=0))
        check(name+'_file_hash',sha(path)==meta['catalog_sha256'])
        print('validated',field,n,flush=True)
    check('complete_unique_parent_conservation',total==prep['n_unique_parent'])
    write_json(OUT/'validation.json',dict(checks=checks,n_checks=len(checks),n_application_rows=total,all_passed=all(checks.values())))
    print('All checks passed:',len(checks),flush=True)

if __name__=='__main__':main()

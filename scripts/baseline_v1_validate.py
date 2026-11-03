#!/usr/bin/env python3
"""Inspect actual frozen experimental and full application products."""
from baseline_v1_common import *
import fitsio
from photdis.stage1 import require_compute

def main():
    require_compute();checks=[]
    def check(name,value):
        passed=bool(value);checks.append(dict(name=name,passed=passed))
        if not passed:raise AssertionError(name)
    sample=json.load(open(OUT/'sample.json'));lock=json.load(open(OUT/'selection.json'));t=load(OLD/'references.npz')
    fixed=load(OUT/'sample.npz');config_hash=sha(ROOT/'configs/baseline_v1.json')
    check('sample:identity_hash',identity_hash(fixed['identity'])==sample['identity_sha256'])
    check('sample:unique_count',len(fixed['identity'])==sample['n']==len(np.unique(fixed['identity'])))
    check('sample:config_hash',sample['config_sha256']==config_hash)
    check('selection:config_hash',lock['config_sha256']==config_hash)
    for name in MODELS:
        check(name+':v0_model_unchanged',sha(OLD/(name+'.joblib'))==sample['v0_models'][name])
        train=json.load(open(OUT/(name+'_train.json')));tier=MODELS[name][1]
        mask=(t['split']==0)&t['fit_eligible']&available(t,tier)
        check(name+':training_identity',train['train_identity_sha256']==identity_hash(t['identity'][mask]))
        check(name+':model_hash',train['model_sha256']==sha(OUT/(name+'.joblib')))
        check(name+':deployable_variant',lock['models'][name]['variant'] in ('wide','natural','balanced'))
        for variant in ('wide','natural','balanced','oracle'):
            p=load(OUT/f'{name}_{variant}_evaluation.npz');rows=p['rows']
            check(name+variant+':only_heldout',np.all(np.isin(t['split'][rows],[1,2])))
            check(name+variant+':no_training_identity',not np.intersect1d(t['identity'][rows],t['identity'][mask]).size)
            check(name+variant+':pdf',np.isfinite(p['pdf']).all() and np.all(p['pdf']>=0) and np.allclose(p['pdf'].sum(axis=1),1,atol=2e-5))
            check(name+variant+':quantiles',np.isfinite(p['mu']).all() and np.all(np.diff(p['mu'],axis=1)>=0))
            check(name+variant+':weights',np.all(p['weights']>=-1e-6) and np.allclose(p['weights'].sum(axis=1),1,atol=2e-5))
    for a,b in [('A_on_B','B'),('A_on_C','C'),('B_on_C','C')]:
        x=json.load(open(OUT/(a+'_train.json')));y=json.load(open(OUT/(b+'_train.json')))
        check(a+':same_training',x['train_identity_sha256']==y['train_identity_sha256'])
        x=load(OUT/f'{a}_wide_evaluation.npz');y=load(OUT/f'{b}_wide_evaluation.npz')
        check(a+':same_evaluation',np.array_equal(x['rows'],y['rows']))
    depth=load(OUT/'depth_input.npz')
    check('depth:test_templates_only',np.all(t['split'][depth['source_row']]==2))
    check('depth:consistent_shift',np.allclose(depth['truth_mu'],truth_mu(t,'chosen')[depth['source_row']]+depth['shift']))
    check('depth:no_identity_cross',not np.intersect1d(depth['identity'],t['identity'][t['split']==0]).size)
    check('depth:actual_input_hash',sha(OUT/'depth_input.npz')==json.load(open(OUT/'depth_input.json'))['input_sha256'])
    depth_result=json.load(open(OUT/'depth_result.json'))
    check('depth:scored_input_hash',depth_result['inputs_sha256']==sha(OUT/'depth_input.npz'))
    check('depth:frozen_selection_hash',depth_result['selection_sha256']==sha(OUT/'selection.json'))
    dp=load(OUT/'depth_predictions.npz')
    check('depth:accepted_predictions',np.isfinite(dp['v1_mu'][depth['accepted']]).all())
    check('depth:rejected_preserved',np.isnan(dp['v1_mu'][~depth['accepted']]).all())
    rec=load(OUT/'recommended_evaluation.npz')
    check('recommended:ordered_calibration',np.all(np.diff(rec['mu_cal'],axis=1)>=0))
    check('recommended:calibration_contains_raw',np.all(rec['mu_cal'][:,[0,1]]<=rec['mu'][:,[0,1]]) and np.all(rec['mu_cal'][:,[3,4]]>=rec['mu'][:,[3,4]]))
    app=json.load(open(OUT/'application.json'));v0app=json.load(open('manifests/baseline_v0_report.json'))['application'];ids=[]
    check('application:frozen_selection_hash',app['selection_sha256']==sha(OUT/'selection.json'))
    for field in range(14):
        path=OUT/f'catalog_{field:02}.fits';old=OLD/f'catalog_{field:02}.fits'
        check(f'field{field}:v0_catalog_unchanged',sha(old)==v0app[field]['catalog_sha256'])
        check(f'field{field}:v1_catalog_hash',sha(path)==app['fields'][field]['sha256'])
        with fitsio.FITS(str(path)) as f,fitsio.FITS(str(old)) as g:
            n=f[1].get_nrows();check(f'field{field}:row_count',n==g[1].get_nrows()==app['fields'][field]['n'])
            valid_identity=True;valid_units=True;valid_state=True;valid_available=True
            valid_weights=True;valid_routing=True;valid_intervals=True;valid_policy=True
            for start in range(0,n,100000):
                rows=np.arange(start,min(start+100000,n));a=f[1].read(rows=rows,columns=[
                    'identity','route','mu','mu_cal','distance_kpc','internal_uncalibrated','version','B','C',
                    'A_mu','B_mu','C_mu','draws','branch_weights','supported','ambiguous','optical_error',
                    'numerical_converged','research_quality','A_weights','B_weights','C_weights',
                    'A_supported','B_supported','C_supported','A_ambiguous','B_ambiguous','C_ambiguous'])
                b=g[1].read(rows=rows,columns=['identity']);ids.append(a['identity'])
                valid_identity &= np.array_equal(a['identity'],b['identity'])
                valid_state &= np.all(a['version']==1)&np.all(a['internal_uncalibrated'])
                if field==13:
                    valid_units &= np.all(a['route']==-1)&np.isnan(a['mu']).all()&np.isnan(a['distance_kpc']).all()
                    valid_policy &= not np.any(a['supported']|a['research_quality']|a['numerical_converged']) and np.all(a['draws']==0)
                    valid_weights &= np.isnan(a['branch_weights']).all()
                    valid_routing &= all(np.isnan(a[k+'_mu']).all() for k in 'ABC')
                    valid_intervals &= np.isnan(a['mu_cal']).all()
                else:
                    valid_units &= np.isfinite(a['mu']).all()&np.all(np.diff(a['mu'],axis=1)>=0)&np.allclose(a['distance_kpc'],10**((a['mu'].astype(float)-10)/5),rtol=3e-6)
                    valid_state &= np.all(np.isin(a['route'],[0,1,2]))&np.all(np.isin(a['draws'],[32,128]))
                    for name in 'BC':valid_available &= np.isfinite(a[name+'_mu'][a[name]]).all()&np.isnan(a[name+'_mu'][~a[name]]).all()
                    for code,name in enumerate('ABC'):
                        use=np.ones(len(a),bool) if name=='A' else a[name]
                        w=a[name+'_weights'][use]
                        valid_weights &= np.isfinite(w).all() and np.all(w>=0) and np.allclose(w.sum(axis=1),1,atol=3e-6)
                        selected=a['route']==code
                        valid_routing &= np.all(use[selected]) and np.array_equal(a['mu'][selected],a[name+'_mu'][selected])
                        valid_routing &= np.array_equal(a['branch_weights'][selected],a[name+'_weights'][selected])
                        valid_routing &= np.array_equal(a['supported'][selected],a[name+'_supported'][selected])
                        valid_routing &= np.array_equal(a['ambiguous'][selected],a[name+'_ambiguous'][selected])
                    valid_intervals &= np.all(np.diff(a['mu_cal'],axis=1)>=0) and np.all(a['mu_cal'][:,[0,1]]<=a['mu'][:,[0,1]]) and np.all(a['mu_cal'][:,[3,4]]>=a['mu'][:,[3,4]])
                    for layer,mask in [('low',a['optical_error']<.03),('high',a['optical_error']>=.03)]:
                        valid_policy &= np.all(a['draws'][mask]==lock['numerical'][layer]['draws'])
                        valid_policy &= np.all(a['numerical_converged'][mask]==lock['numerical'][layer]['converged'])
                    if lock['quality_rule'] is None:valid_policy &= not np.any(a['research_quality'])
            check(f'field{field}:identity_order',valid_identity);check(f'field{field}:distance_units',valid_units)
            check(f'field{field}:state',valid_state);check(f'field{field}:availability',valid_available)
            check(f'field{field}:weights_or_no_model',valid_weights);check(f'field{field}:selected_model',valid_routing)
            check(f'field{field}:expanded_intervals',valid_intervals);check(f'field{field}:frozen_policy_flags',valid_policy)
        print('Validated',field,n,flush=True)
    ids=np.concatenate(ids);check('complete_parent_count',len(ids)==9390518);check('unique_parent',len(np.unique(ids))==len(ids))
    write_json(OUT/'validation.json',dict(n_checks=len(checks),all_passed=all(c['passed'] for c in checks),checks=checks,n_application_rows=len(ids),application_sha256=sha(OUT/'application.json')))
    print('All actual checks passed',len(checks),flush=True)
if __name__=='__main__':main()

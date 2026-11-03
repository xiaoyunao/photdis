#!/usr/bin/env python3
"""Validate real v1.1 products and the frozen baseline provenance."""
from baseline_v1_1_common import *
from baseline_v1_1_score import adoption
from photdis.stage1 import require_compute
import joblib


def main():
    require_compute();checks=[]
    def check(name,value):
        passed=bool(value);checks.append(dict(name=name,passed=passed))
        if not passed:raise AssertionError(name)
    t=load(OLD/'references.npz');lock=json.load(open(OUT/'selection.json'))
    for name in MODELS:
        info=json.load(open(OUT/f'{name}_train.json'));e=load(OUT/f'{name}_evidence.npz');rows,weight,fit=evidence_rows(t,name)
        check(name+':crossfit_rows',np.array_equal(e['rows'],rows))
        check(name+':inclusion_weights',np.array_equal(e['sample_weight'],weight))
        check(name+':training_hash',info['train_identity_sha256']==identity_hash(t['identity'][fit]))
        check(name+':evidence_hash',info['evidence_sha256']==sha(OUT/f'{name}_evidence.npz'))
        check(name+':gate_hash',info['gate_sha256']==sha(OUT/f'{name}_gate.joblib'))
        check(name+':frozen_expert_hash',info['frozen_expert_sha256']==sha(EXPERT/f'{name}.joblib'))
        oldinfo=json.load(open(EXPERT/f'{name}_train.json'))
        check(name+':original_v1_expert',oldinfo['model_sha256']==sha(EXPERT/f'{name}.joblib'))
        check(name+':config_hash',info['config_sha256']==sha(ROOT/'configs/baseline_v1_1.json'))
        check(name+':nonnegative_finite_evidence',np.all(np.isfinite(e['effective'])) and np.all(e['effective']>=0))
        for fold,fields in enumerate(CFG['inner_folds']):
            f=load(OUT/f'{name}_fold{fold}_identities.npz');tr=f['train_rows'];hold=f['held_rows']
            check(f'{name}:fold{fold}:train_only',np.all(t['split'][tr]==0) and np.all(t['split'][hold]==0))
            check(f'{name}:fold{fold}:spatial_isolation',np.all(np.isin(t['field'][hold],fields)) and not np.any(np.isin(t['field'][tr],fields)))
            check(f'{name}:fold{fold}:identity_isolation',not np.intersect1d(t['identity'][hold],t['identity'][tr]).size)
            check(f'{name}:fold{fold}:actual_evidence',np.array_equal(hold,e['rows'][e['fold']==fold]))
        check(name+':no_oracle_deployment',lock['models'][name]['variant'] in ('wide','evidence'))
        for variant in VARIANTS:
            p=load(OUT/f'{name}_{variant}_evaluation.npz');r=p['rows']
            check(name+variant+':heldout_identity',np.array_equal(r,reference_rows(t,name)) and np.array_equal(p['identity'],t['identity'][r]))
            check(name+variant+':pdf',np.all(np.isfinite(p['pdf'])) and np.all(p['pdf']>=0) and np.allclose(p['pdf'].sum(axis=1),1,atol=2e-5))
            check(name+variant+':weights',np.all(p['weights']>=-1e-6) and np.allclose(p['weights'].sum(axis=1),1,atol=2e-5))
            check(name+variant+':quantiles',np.all(np.isfinite(p['mu'])) and np.all(np.diff(p['mu'],axis=1)>=0))
            if variant in ('wide','natural','balanced','oracle'):
                old=load(EXPERT/f'{name}_{variant}_evaluation.npz');pos=np.searchsorted(old['rows'],r)
                check(name+variant+':v1_identical_rows',np.array_equal(old['rows'][pos],r))
                check(name+variant+':v1_reproduction',np.allclose(old['mu'][pos],p['mu'],atol=3e-5,rtol=0))
        check(name+':adoption_guards',lock['models'][name]['adopt']==all(lock['models'][name]['checks'].values()))
    for a,b in [('A_on_B','B'),('A_on_C','C'),('B_on_C','C')]:
        x=load(OUT/f'{a}_evidence.npz');y=load(OUT/f'{b}_evidence.npz')
        check(a+':same_evidence',np.array_equal(x['identity'],y['identity']) and np.array_equal(x['sample_weight'],y['sample_weight']))
        check(a+':same_folds',np.array_equal(x['fold'],y['fold']))
    samples=json.load(open(OUT/'samples.json'))
    for kind in ('parent','parallax'):
        sample=load(OUT/f'{kind}_sample.npz')
        check(kind+':sample_hash',sha(OUT/f'{kind}_sample.npz')==samples[kind+'_sha256'])
        check(kind+':identity',identity_hash(sample['identity'])==samples[kind]['identity_sha256'])
        check(kind+':unique',len(sample['identity'])==len(np.unique(sample['identity'])))
        if kind=='parallax':
            check('parallax:test_only',np.all(sample['split']==2))
            check('parallax:negative_retained',np.sum(sample['p_corr']<0)==samples[kind]['negative'] and samples[kind]['negative']>0)
        for name in 'ABC':
            for variant in ('wide','evidence'):
                p=load(OUT/f'{kind}_{name}_{variant}.npz')
                check(kind+name+variant+':identity',np.array_equal(p['identity'],sample['identity'][p['rows']]))
                check(kind+name+variant+':pdf',np.all(np.isfinite(p['pdf'])) and np.allclose(p['pdf'].sum(axis=1),1,atol=2e-5))
    for variant in ('wide','evidence','selected'):
        p=load(OUT/f'routed_{variant}_evaluation.npz');wide=load(OUT/'routed_wide_evaluation.npz')
        check('route:'+variant,np.array_equal(p['route'],wide['route']))
    # Hash old artifacts rather than invoking any old training or application.
    app=json.load(open(ROOT/'manifests/baseline_v1_application.json'))
    check('v1:application_manifest',sha(EXPERT/'application.json')==sha(ROOT/'manifests/baseline_v1_application.json'))
    for f in app['fields']:check(f"v1:catalog{f['field']}:unchanged",sha(EXPERT/f"catalog_{f['field']:02}.fits")==f['sha256'])
    hashes={p.name:sha(p) for p in OUT.glob('*.json') if p.name!='validation.json'}
    write_json(OUT/'validation.json',dict(n_checks=len(checks),all_passed=True,checks=checks,product_hashes=hashes))
    print('Actual validation checks passed',len(checks),flush=True)
if __name__=='__main__':main()

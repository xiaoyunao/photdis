#!/usr/bin/env python3
"""Acceptance on actual saved products, not synthetic success declarations."""
from baseline_v1_2_run import *
from photdis.baseline_v1_2 import conservative_threshold,single_mask,nominal_old_weights


def main():
    require_compute();t=load(OLD/'references.npz');checks={};hashes={};ids={};reproduction={}
    def check(key,value):
        checks[key]=bool(value)
        if not value:raise AssertionError(key)
    for name,(_,tier) in MODELS.items():
        tr=load(OUT/f'{name}_train.npz');fit=json.load(open(OUT/f'{name}_fit.json'));r=tr['rows'];ids[name]=tr['identity']
        expected=np.flatnonzero((t['split']==0)&t['fit_eligible']&(t['class_label']>=0)&available(t,tier))
        check(name+':training_rows',np.array_equal(r,expected))
        check(name+':training_identity',np.array_equal(tr['identity'],t['identity'][r]))
        check(name+':balanced',np.allclose([tr['weight'][t['class_label'][r]==k].sum() for k in range(3)],len(r)/3))
        check(name+':classifiers_same_ids',fit['quadratic']['identity_sha256']==fit['forest']['identity_sha256'])
        check(name+':classifiers_same_weights',fit['quadratic']['weight_sha256']==fit['forest']['weight_sha256'])
        check(name+':quadratic_converged',max(fit['quadratic']['iterations'])<2000)
        check(name+':frozen_expert',fit['expert_sha256']==sha(EXPERT/f'{name}.joblib'))
        scores=json.load(open(OUT/f'{name}_scores.json'));model=joblib.load(EXPERT/f'{name}.joblib')
        for kind in (('reference','parent','parallax') if name in 'ABC' else ('reference',)):
            table=t if kind=='reference' else load(PREV/f'{kind}_sample.npz')
            cache=load(OUT/f'{kind}_{name}_cache.npz');rows=cache['rows']
            check(f'{kind}_{name}:identity',np.array_equal(cache['identity'],table['identity'][rows]))
            check(f'{kind}_{name}:cache_mass',np.max(abs(cache['branch_pdf'].sum(axis=2)-1))<2e-6)
            if kind=='reference':check(name+':heldout',not np.intersect1d(cache['identity'],tr['identity']).size)
            for variant in VARS:
                tag=f'{kind}_{name}_{variant}';p=load(OUT/f'{tag}.npz')
                check(tag+':same_rows',np.array_equal(p['rows'],rows))
                check(tag+':mass',np.max(abs(p['pdf'].sum(axis=1)-1))<2e-6)
                check(tag+':weights',np.allclose(p['weights'].sum(axis=1),1,atol=1e-7))
                check(tag+':raw_weights',np.allclose(p['raw_weights'].sum(axis=1),1,atol=1e-7))
                check(tag+':support',not np.any(p['weights'][:,:3][~p['branch_support']]>1e-12))
                check(tag+':main_photometric',np.array_equal(p['main_index'],p['weights'].argmax(axis=1)))
                main=cache['branch_pdf'][np.arange(len(rows)),p['main_index']]
                check(tag+':main_quantiles',np.allclose(p['main_mu'],distance_summary(main,p['edges'],table['g0'][rows]),atol=1e-8))
                check(tag+':ordered',np.all(np.diff(p['mu'],axis=1)>=-1e-10) and np.all(np.diff(p['main_mu'],axis=1)>=-1e-10))
                if kind=='reference' and variant in ('natural','balanced','evidence'):
                    old=load(PREV/f'{name}_{variant}_evaluation.npz')
                    diff=float(np.max(abs(np.cumsum(p['pdf'],axis=1,dtype=float)-np.cumsum(old['pdf'],axis=1,dtype=float))))
                    qdiff=float(np.max(abs(p['mu']-old['mu'])))
                    reproduction[name+'_'+variant]=dict(max_cdf=diff,max_mu=qdiff)
                    check(tag+':frozen_cdf',diff<=1e-6);check(tag+':frozen_quantiles',qdiff<=.001)
                if kind=='reference':
                    label=t['class_label'][rows];cal=t['split'][rows]==1
                    nominal=nominal_old_weights(model,arrays(t,rows)[0],variant) if variant in ('natural','balanced') else p['raw_weights']
                    for key,v in scores[variant]['working_points'].items():
                        raw=p['weights' if v['score_state']=='final' else 'raw_weights'][:,v['target']]
                        if v['score_state']=='nominal' and variant in ('natural','balanced'):
                            raw=nominal[:,v['target']]
                        threshold=conservative_threshold(raw[cal&(label==0)],v['budget'])
                        actual=v['threshold'] if v['threshold'] is not None else np.inf
                        check(name+variant+key+':threshold_cal_only',threshold==actual)
                        check(name+variant+key+':cal_budget',(v['cal']['dwarf_fpr'] or 0)<=v['budget']+1e-12)
                        check(name+variant+key+':all_contaminants',sum(v['test']['class_counts'].values())==v['test']['n_selected'])
        del cache
    for names in [('B','A_on_B'),('C','A_on_C','B_on_C')]:
        for name in names[1:]:check(name+':same_training',np.array_equal(ids[name],ids[names[0]]))
    par=json.load(open(OUT/'parallax.json'))
    for name,variants in par['results'].items():
        for variant,entry in variants.items():
            c=entry['checks'];tag=name+variant
            check(tag+':bin_mass',c['bin_mass_max_error']<2e-6)
            check(tag+':actual_bin',c['observed_bin_mass_max_error']<1e-12)
            check(tag+':density_factorization',c['density_factorization_max_relative_error']<1e-10)
            check(tag+':quadrature',c['quadrature_full_pit_max_difference']<1e-3)
            check(tag+':no_faint_claim',entry['groups']['r22_25']['n']==0)
    check('Gaia_negative_preserved',par['results']['A']['wide_mixture']['groups']['all']['n_negative']==1800)
    samples=json.load(open(PREV/'samples.json'))
    for kind in ('parent','parallax'):
        check(kind+':frozen_sample',sha(PREV/f'{kind}_sample.npz')==samples[kind+'_sha256'])
    app=json.load(open(ROOT/'manifests/baseline_v1_application.json'))
    check('v1:application_manifest',sha(EXPERT/'application.json')==sha(ROOT/'manifests/baseline_v1_application.json'))
    for f in app['fields']:
        check(f"v1:catalog{f['field']}:unchanged",sha(EXPERT/f"catalog_{f['field']:02}.fits")==f['sha256'])
    for p in sorted(OUT.iterdir()):
        if p.suffix in ('.npz','.joblib','.json') and p.name not in ('validation.json','report.json'):hashes[p.name]=sha(p)
    hashes['config']=sha(ROOT/'configs/baseline_v1_2.json')
    hashes['parent_sample']=sha(PREV/'parent_sample.npz');hashes['Gaia_sample']=sha(PREV/'parallax_sample.npz')
    write_json(OUT/'validation.json',dict(n_checks=len(checks),all_passed=all(checks.values()),checks=checks,product_hashes=hashes,reproduction=reproduction))
    print('Actual validation',len(checks),'passed',flush=True)
if __name__=='__main__':main()

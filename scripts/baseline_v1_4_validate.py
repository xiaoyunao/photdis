#!/usr/bin/env python3
"""Allocated artifact/identity checks after all fixed v1.4 experiments."""
from baseline_v1_4_experiments import *


def main():
    require_compute();checks={}
    assembly=json.loads((OUT/'assembly.json').read_text())
    eligibility=json.loads((OUT/'eligibility.json').read_text())
    checks['eligibility_reference_hash']=eligibility['reference_sha256']==assembly['reference_sha256']
    zero_vac=json.loads((OUT/'zero_vac_quality_audit.json').read_text())
    for kind,v in zero_vac.items():
        checks[kind+':zero_vac_inventory']=v['zero_id_rows']==assembly['vac_audit'][kind]['zero_or_invalid_id_rows']
        checks[kind+':no_qualified_zero_vac_lost']=v['quality_passing']==0
    checks.update({'assembly:'+k:bool(v) for k,v in assembly['checks'].items()})
    checks['reference_hash']=sha(OUT/'references.npz')==assembly['reference_sha256']
    with np.load(OUT/'references.npz') as z:
        t={k:z[k] for k in ['identity','ls_id','split','class_eligible','class_train_eligible','identity_conflict','nonstellar_risk','photometric_eligible','class_label','fit_eligible','label_source','Mg','optical_error','i_usable','match_epoch_known','match_separation_arcsec','zero_id_match','historical_identity']}
    checks['class_independent']=np.array_equal(t['class_train_eligible'],t['class_eligible'])
    checks['class_qualified']=np.array_equal(t['class_eligible'],t['photometric_eligible']&~t['identity_conflict']&~t['nonstellar_risk']&(t['class_label']>=0))
    checks['distance_precise']=bool(np.all((t['label_source'][t['fit_eligible']]>0)&(t['optical_error'][t['fit_eligible']]<.03)&np.isfinite(t['Mg'][t['fit_eligible']])))
    for sp in [0,1,2]:
        checks[f'no_historical_split{sp}']=not bool(np.any((t['split']==sp)&t['historical_identity']))
    zero=t['zero_id_match'];threshold=np.where(t['match_epoch_known'],1.,.5)
    checks['zero_separation']=bool(np.all(t['match_separation_arcsec'][zero]<=threshold[zero]))
    e=OUT/'experiments';models={}
    for stage,names in [('frozen','ABC'),('expanded','ABC'),('matched',['grz','griz'])]:
        completion=json.loads((e/(stage+'_complete.json')).read_text())
        checks[stage+':reference_hash']=completion['reference_sha256']==assembly['reference_sha256']
        for name in names:
            prefix=stage+'_'+name;ids=load(e/(prefix+'_identities.npz'))
            for kind in ['class','distance']:
                train=ids[kind+'_train_identity'];ev=ids[kind+'_eval_identity']
                checks[prefix+':'+kind+':unique_train']=len(np.unique(train))==len(train)
                checks[prefix+':'+kind+':no_leak']=not bool(np.intersect1d(train,ev).size)
                checks[prefix+':'+kind+':unique_eval']=len(np.unique(ev))==len(ev)
                if stage=='expanded':
                    legacy=load(e/(prefix+'_legacy_training.npz'))[kind+'_train_identity']
                    checks[prefix+':'+kind+':legacy_no_leak']=not bool(np.intersect1d(legacy,ev).size)
                    checks[prefix+':'+kind+':new_is_added']=not bool(np.intersect1d(legacy,train).size)
            for kind in ['classifier','distance']:
                p=e/(prefix+'_'+kind+'.joblib');m=json.loads(p.with_suffix('.json').read_text())
                checks[prefix+':'+kind+':hash']=sha(p)==m['sha256']
            p=e/(prefix+'_distance');score=json.loads((p/'scores.json').read_text())
            checks[prefix+':evaluation_hash']=sha(p/'evaluation.npz')==score['evaluation_sha256']
            for k,v in score['checks'].items():
                if isinstance(v,bool):checks[prefix+':'+k]=v
            for marker in sorted(p.glob('[0-9]*.json')):
                checks[prefix+':chunk:'+marker.stem]=sha(marker.with_suffix('.npz'))==json.loads(marker.read_text())['sha256']
            integrity=json.loads((e/(prefix+'_class_integrity.json')).read_text())
            checks[prefix+':class_score_hash']=sha(e/(prefix+'_class_scores.npz'))==integrity['sha256']
            checks[prefix+':class_score_reference']=integrity['reference_sha256']==assembly['reference_sha256']
            classification=json.loads((e/(prefix+'_classification.json')).read_text())
            for variant,v in classification.items():
                checks[prefix+':'+variant+':cal_budget']=v['cal_dwarf_fpr'] is None or v['cal_dwarf_fpr']<=.02+1e-12
            refined=OUT/'gaia_refined'/prefix
            gr=json.loads((refined/'scores.json').read_text())
            checks[prefix+':refined_reference']=gr['fingerprint']['reference_sha256']==assembly['reference_sha256']
            checks[prefix+':refined_primary']=gr['fingerprint']['primary_score_sha256']==sha(p/'scores.json')
            checks[prefix+':refined_model']=gr['fingerprint']['model_sha256']==sha(e/(prefix+'_distance.joblib'))
            checks[prefix+':refined_evaluation']=gr['evaluation_sha256']==sha(refined/'evaluation.npz')
            for k,v in gr['checks'].items():
                if isinstance(v,bool):checks[prefix+':refined:'+k]=v
            checks[prefix+':refined_negative_count']=gr['checks']['negative_retained']==score['checks']['gaia_negative_retained']
            for marker in sorted(refined.glob('[0-9]*.json')):
                content=json.loads(marker.read_text())
                checks[prefix+':refined_chunk:'+marker.stem]=sha(marker.with_suffix('.npz'))==content['sha256'] and sha(p/marker.with_suffix('.npz').name)==content['primary_sha256']
            for group,values in gr['groups'].items():
                checks[prefix+':refined_n:'+group]=values['n']==score['groups'][group]['full_Gaia']['n']
                score['groups'][group]['full_Gaia'].update(values)
            models[prefix]=dict(classification=classification,distance=score,
                training=completion['models'][name]['metadata'],gaia_refinement=gr['numerical_change'])
    for name in 'ABC':
        f=load(e/('frozen_'+name+'_identities.npz'));x=load(e/('expanded_'+name+'_identities.npz'))
        for kind in ['class','distance']:checks[name+':same_eval_'+kind]=np.array_equal(f[kind+'_eval_identity'],x[kind+'_eval_identity'])
    a=load(e/'matched_grz_identities.npz');b=load(e/'matched_griz_identities.npz')
    for k,v in a.items():checks['matched:'+k]=np.array_equal(v,b[k])
    report=dict(n_checks=len(checks),all_passed=all(checks.values()),checks=checks,
        limitation='Artifact integrity and numerical/identity consistency do not establish absolute astrophysical accuracy.')
    save_json(OUT/'validation.json',report)
    assert report['all_passed'],[k for k,v in checks.items() if not v]
    # Compact report includes inspectable scientific group metrics, omitting huge
    # per-chunk validation maps and per-cell classifications from the summary.
    compact={}
    for name,m in models.items():
        compact[name]=dict(classification={variant:dict(v,groups={k:g for k,g in v['groups'].items() if not k.startswith('cell')}) for variant,v in m['classification'].items()},
            distance={k:v for k,v in m['distance']['groups'].items() if not k.startswith('cell')},n_distance=m['distance']['n'],
            numerical_checks=m['distance']['checks'],gaia_refinement=m['gaia_refinement'],training=m['training'])
    save_json(OUT/'result.json',dict(assembly=assembly,eligibility=eligibility,zero_vac_quality=zero_vac,parent_coverage=json.loads((OUT/'parent_coverage.json').read_text()),
        models=compact,validation=dict(n_checks=len(checks),all_passed=True),
        deployment='Preserve v1. Fixed validation only; no model search or full-catalog rerun.',
        validation_limits='New geometric reference-selected sky.22–25 claims require actual truth counts; unrepresented cells remain unvalidated.'))
    print('VALIDATED',len(checks),flush=True)

if __name__=='__main__':main()

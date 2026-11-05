#!/usr/bin/env python3
"""Audit normalized report views and bind matching/interval provenance."""
import csv
import json
import numpy as np
from v2_0_common import *
from photdis.stage1 import require_compute

CHECKS={}
def check(name,value):
    CHECKS[name]=bool(value)
    assert value,name


def main():
    require_compute()
    snap=json.loads((TABLES/'snapshot.json').read_text())
    with np.load(OUT/'references.npz') as z:
        provenance={k:z[k] for k in ('identity','specdis_source_only_join','spdist_source_only_join',
                                    'specdis_row','spdist_row','rvs_quality_reason','targetid')}
    for item in snap['models']:
        name=item['model']
        q=load(DEST/f'chosen_{name}.npz')
        check(name+'_unique',len(np.unique(q['identity']))==len(q['identity']))
        check(name+'_training_disjoint',not q['in_model_training'].any())
        check(name+'_model_id',np.all(q['model_id']==name))
        if not name.startswith('baseline_v1_'):
            index=unique_join(q['identity'],provenance['identity'])
            check(name+'_source_join',(index>=0).all())
            for key in provenance:
                if key!='identity':
                    q[key]=provenance[key][index]
        elif 'desi_object_id' in q:
            q['targetid_text']=q['desi_object_id']
        if 'pdf_source_row' not in q:
            suffix='recommended_evaluation.npz' if name=='baseline_v1_recommended' else name.removeprefix('baseline_v1_')+'_wide_evaluation.npz'
            original=ROOT/'data/baseline_v1'/suffix
            with np.load(original) as z:
                q['pdf_source_row']=z['rows']
                q['pdf_evaluation_row']=np.arange(len(q['identity']))
                q['cached_full_pdf']=np.full(len(q['identity']),'pdf' in z.files)
            item['original_evaluation']=str(original.relative_to(ROOT))
            item['full_pdf_location']='original evaluation PDF indexed by pdf_evaluation_row, if cached_full_pdf; otherwise frozen model reconstruction'
        check(name+'_ordered',np.all(np.diff(q['mu'],axis=1)>=0))
        check(name+'_raw_quantiles',np.array_equal(q['mu'],q.get('mu_raw',q['mu'])))
        for source in ('chosen','Gaia','specdis','spdist'):
            expected=100*np.expm1(np.log(10)*(q['mu'][:,2]-q['ref_'+source])/5)
            check(name+'_'+source+'_epsilon',np.allclose(expected,q['epsilon_pct_'+source],equal_nan=True))
        atomic_npz(DEST/f'chosen_{name}.npz',**q)
        sp=DEST/f'sources_{name}.npz'
        source_view=load(sp)
        link=source_view['chosen_view_row']
        check(name+'_source_identity',np.array_equal(source_view['identity'],q['identity'][link]))
        key=np.rec.fromarrays([source_view['identity'],source_view['source']])
        check(name+'_source_unique',len(np.unique(key))==len(key))
        path=TABLES/f'performance_{name}.csv'
        with open(path) as f:rows=[r for r in csv.DictReader(f) if r.get('interval')!='calibrated_interval']
        for record in rows:
            record['interval']='raw'
        if 'mu_cal' in q:
            for split,label in ((1,'cal'),(2,'development_test')):
                for group,mask in strata(q).items():
                    use=mask&(q['split']==split)
                    for kind in ('chosen','Gaia','specdis','spdist'):
                        rows.append(dict(model=name,split=label,group=group,reference=kind,interval='calibrated_interval',
                                         **percent_metrics(q['mu_cal'][use],q['ref_'+kind][use])))
        csv_rows(path,rows)
        item.update(chosen_view_sha256=sha(DEST/f'chosen_{name}.npz'),
                    source_view=str(sp.relative_to(ROOT)),source_view_sha256=sha(sp),
                    training_file_sha256=sha(ROOT/item['training_identity_file']))
    q=load(DEST/'chosen_expanded_A.npz')
    test=q['split']==2
    check('chosen47756',np.sum(test&np.isfinite(q['ref_chosen']))==47756)
    check('PSF47740',np.sum(test&np.isfinite(q['ref_chosen'])&(q['TYPE']=='PSF'))==47740)
    check('Gaia49327',np.sum(test&q['parallax_eligible'])==49327)
    check('negative4139',np.sum(test&q['parallax_eligible']&(q['p_corr']<0))==4139)
    registry=json.loads((TABLES/'figure_registry.json').read_text())
    check('figure_families',all(any(name.startswith(f'{i:02d}') for name,r in registry.items() if r['section']=='main') for i in range(1,11)))
    for name,record in registry.items():
        for suffix in ('png','pdf'):
            actual=FIG/record['section']/(name+'.'+suffix)
            check(name+'_'+suffix,sha(actual)==record[suffix+'_sha256'])
        check(name+'_csv',(TABLES/'figures'/(name+'.csv')).is_file())
        record['snapshot_frozen_model']=CFG['baseline_commit']
    snap['report_schema']=dict(intervals='raw point-reference agreement versus cal-expanded application intervals',
                              chosen='one object/model row',sources='normalized FK view, multiple sources per star',
                              pdf='explicit availability, source row or cached chunk locator',
                              unknown='class_label=-1, kept separate')
    snap['report_code_hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/v2_0_figures.py',ROOT/'scripts/v2_0_finalize.py']})
    save_json(TABLES/'snapshot.json',snap)
    save_json(TABLES/'figure_registry.json',registry)
    save_json(TABLES/'validation.json',dict(checks=CHECKS,n=len(CHECKS)))
    save_json(DEST/'collection_complete.json',dict(snapshot_sha256=sha(TABLES/'snapshot.json')))
    print('COMPLETE final view audit',len(CHECKS),'checks',flush=True)


if __name__=='__main__':
    main()

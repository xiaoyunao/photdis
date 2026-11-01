#!/usr/bin/env python3
"""Fixed application/low-FPR/abundance contrasts on existing spectral records."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'.deps')]
import joblib
import numpy as np
from photdis.giant_information import (BANDS, ERRORS, DIMENSIONS, fit_classifier,
    predict_classifier, fit_abundance, predict_abundance, abundance_metrics)
from photdis.giant_candidate import (observable_masks, selection_summary,
    threshold_at_fpr, feasibility, mechanism_features)
from giant_information_fit import basic_selection
from photdis.stage1 import require_compute, write_json
CONFIG = Path('configs/giant_candidate_v1.json')


def fit_possible(t, rows, cfg):
    return ((t['label'][rows]==1).sum()>=cfg['min_train_giants']
            and (t['label'][rows]==0).sum()>=cfg['min_train_dwarfs']
            and len(np.unique(t['field_index'][rows]))>=cfg['min_train_fields'])


def assert_disjoint(t, train, test):
    assert not set(t['identity_group'][train]) & set(t['identity_group'][test])
    assert not set(t['field_index'][train]) & set(t['field_index'][test])


def fit_one(t, bands, errors, train, name, cfg, seed):
    return fit_classifier(bands[train], errors[train], t['label'][train], DIMENSIONS[name],
                          C=cfg['logistic_C'], draws=cfg['band_draws'], seed=seed)


def predict(model, bands, errors, rows, cfg, seed):
    # Bound temporary Monte Carlo arrays for the wider spectral sample.
    chunks = [predict_classifier(model, bands[r], errors[r], draws=cfg['band_draws'], seed=seed+j)
              for j,r in enumerate(np.array_split(rows, max(1, (len(rows)+9999)//10000)))]
    return tuple(np.concatenate([c[k] for c in chunks]) for k in range(3))


def inner_thresholds(t, bands, errors, base, training, cfg, new, outer, out):
    fields = np.unique(t['field_index'][np.flatnonzero(base & (t['field_index']!=outer))])
    outputs = {name: [] for name in ('A','B')}
    logs = []
    for group in range(new['inner_field_groups']):
        held_fields = fields[group::new['inner_field_groups']]
        test = np.flatnonzero(base & np.isin(t['field_index'],held_fields))
        train = np.flatnonzero(training & (t['field_index']!=outer) & ~np.isin(t['field_index'],held_fields))
        assert_disjoint(t,train,test)
        assert outer not in t['field_index'][train] and outer not in t['field_index'][test]
        log={'held_fields':held_fields.tolist(),'training_fields':np.unique(t['field_index'][train]).tolist(),
             'n_train':len(train),'n_test':len(test),'scored':False}
        if len(test) and fit_possible(t,train,cfg):
            log['scored']=True
            for name in outputs:
                model=fit_one(t,bands,errors,train,name,cfg,new['seed']+outer*10+group)
                score,_,outside=predict(model,bands,errors,test,cfg,new['seed']+1000+outer*10+group)
                dwarf=t['label'][test]==0
                outputs[name].append(np.where(outside[dwarf],-1.,score[dwarf]))
        logs.append(log)
    complete=all(x['scored'] for x in logs)
    result={name:{str(a):threshold_at_fpr(np.concatenate(parts) if parts and complete else [],a)
                  for a in new['target_fpr']} for name,parts in outputs.items()}
    return result,logs


def main():
    require_compute()
    new=json.loads(CONFIG.read_text()); cfg=json.loads(Path(new['parent_config']).read_text())
    parent=Path(new['parent_directory']); out=Path(new['output_directory']);out.mkdir(parents=True,exist_ok=True)
    resultpath=out/'results.json'
    if resultpath.exists(): raise FileExistsError(resultpath)
    old=json.loads((parent/'results.json').read_text())
    with np.load(parent/'labels.npz',allow_pickle=False) as f: t={k:f[k] for k in f.files}
    bands=np.column_stack([t[k] for k in BANDS]); errors=np.column_stack([t[k] for k in ERRORS])
    masks=observable_masks(t,cfg); training=basic_selection(t,cfg)
    records=[]; mechanism=[]
    for field in sorted(np.unique(t['field_index'][masks['optical']])):
        field=int(field)
        for panel,names in [('optical',['A']),('common5',['A','B','C']),('w1',['A','B'])]:
            rows=np.flatnonzero(masks[panel] & (t['field_index']==field))
            train=np.flatnonzero(training & masks[panel] & (t['field_index']!=field))
            assert_disjoint(t,train,rows)
            rec={'field_index':field,'panel':panel,'n_reference':len(rows),'n_train':len(train),
                 'training_fields':np.unique(t['field_index'][train]).tolist(),'models':{}}
            if not len(rows):
                records.append(rec);continue
            saved={'rows':rows,'identity_group':t['identity_group'][rows]}
            if panel=='w1':
                thresholds,logs=inner_thresholds(t,bands,errors,masks[panel],training & masks[panel],cfg,new,field,out)
                rec['inner_splits']=logs
            for name in names:
                if panel=='w1':
                    if not fit_possible(t,train,cfg):continue
                    model=fit_one(t,bands,errors,train,name,cfg,cfg['seed']+field)
                    joblib.dump(model,out/f'{field}_w1_{name}.joblib')
                else:
                    path=parent/f'DECam_{field}_{panel}_class_{name}.joblib'
                    if not path.exists():continue
                    model=joblib.load(path)
                score,sd,outside=predict(model,bands,errors,rows,cfg,cfg['seed']+100+field)
                if panel!='w1':
                    with np.load(parent/f'DECam_{field}_{panel}_scores.npz') as original:
                        positions=np.searchsorted(rows,original['test_rows'])
                        assert np.array_equal(rows[positions],original['test_rows'])
                        score[positions]=original['score_'+name]
                        sd[positions]=original['score_noise_sd_'+name]
                        outside[positions]=original['color_extrapolation_'+name]
                item={'fixed_half':selection_summary(t,rows,score,outside,masks['common5'],.5),
                      'inside_half':selection_summary(t,rows,score,outside,masks['common5'],.5,True),
                      'descriptive_outer_roc_points':feasibility(t,rows,score,outside,new['target_fpr'])}
                if panel=='w1':
                    item['nested_thresholds']=thresholds[name]; item['nested_results']={}
                    for alpha,rule in thresholds[name].items():
                        if rule['threshold'] is not None:
                            item['nested_results'][alpha]=selection_summary(t,rows,score,outside,masks['common5'],rule['threshold'],True)
                rec['models'][name]=item
                saved['score_'+name]=score;saved['noise_'+name]=sd;saved['outside_'+name]=outside
            np.savez_compressed(out/f'{field}_{panel}_scores.npz',**saved)
            records.append(rec)
            write_json(out/'progress.json',{'completed_application_cells':len(records),'field':field,'panel':panel})
            print(json.dumps({'field':field,'panel':panel,'n':len(rows),'models':list(rec['models'])}),flush=True)
        # Preserve the exact previous abundance masks and field IDs.
        for oldrec in [r for r in old['folds'] if r['field_index']==field and r['system']=='DECam' and r['abundance']]:
            panel=oldrec['panel']; selected=training & masks[panel]
            valid=(t['abundance_calibration_valid'] & (t['label']==1) & (t['desi_feh_err']>0)
                   & (t['desi_feh_err']<=cfg['feh_sigma_max']) & np.isfinite(t['feh_corrected']) & np.isfinite(t['feh_sigma']))
            train=np.flatnonzero(selected & valid & (t['field_index']!=field))
            test=np.flatnonzero(selected & valid & (t['field_index']==field));assert_disjoint(t,train,test)
            with np.load(parent/f'DECam_{field}_{panel}_scores.npz') as original:
                assert np.array_equal(test,original['abundance_test_rows'])
            rec={'field_index':field,'panel':panel,'n_train':len(train),'n_test':len(test),'models':{},'training_fields':np.unique(t['field_index'][train]).tolist()}
            saved={'rows':test,'identity_group':t['identity_group'][test]}
            for name in ['spec']+(['A'] if panel=='optical' else ['A','B','C']):
                features,cov=mechanism_features(t,bands,errors,0 if name=='spec' else DIMENSIONS[name])
                model=fit_abundance(features[train],cov[train],t['feh_corrected'][train],t['feh_sigma'][train],ridge=cfg['feh_ridge'],floor=cfg['feh_intrinsic_floor'])
                mean,sigma=predict_abundance(model,features[test],cov[test])
                metrics,logp=abundance_metrics(t['feh_corrected'][test],t['feh_sigma'][test],mean,sigma)
                rec['models'][name]=metrics;saved['logp_'+name]=logp;saved['mean_'+name]=mean;saved['sigma_'+name]=sigma
                joblib.dump(model,out/f'{field}_{panel}_mechanism_{name}.joblib')
                if name!='spec':rec['models'][name]['gain_vs_spec']=float(np.mean(logp-saved['logp_spec']))
            np.savez_compressed(out/f'{field}_{panel}_mechanism_scores.npz',**saved);mechanism.append(rec)
    write_json(resultpath,{'config':new,'parent_config':cfg,'application':records,'mechanism':mechanism,
        'observed_reference_counts':{k:int(v.sum()) for k,v in masks.items()},
        'inventory':old['inventory'],
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG,Path(new['parent_config']),Path(__file__),ROOT/'src/photdis/giant_candidate.py',parent/'labels.npz',parent/'results.json']}})


if __name__=='__main__':main()

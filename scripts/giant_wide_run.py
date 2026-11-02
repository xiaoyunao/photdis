#!/usr/bin/env python3
"""Predeclared paired fits, same frozen-model calibration and held-field scores."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import joblib
import numpy as np
from photdis.giant_information import BANDS, ERRORS, band_realizations, colors_and_covariance
from photdis.giant_candidate import observable_masks, selection_summary, threshold_at_fpr
from photdis.giant_wide import preprocessing, fit_controlled, field_roles
from photdis.stage1 import require_compute
from giant_candidate_run import assert_disjoint, predict
OUT=Path('data/giant_wide_v1')


def main():
    require_compute(); OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'results.json').exists(): raise FileExistsError(OUT/'results.json')
    cfg=json.loads(Path('configs/giant_information_v1.json').read_text())
    with np.load('data/giant_information_v1/labels.npz') as f: t={k:f[k] for k in f.files}
    masks=observable_masks(t,cfg); base=masks['w1']
    bands=np.column_stack([t[k] for k in BANDS]); errors=np.column_stack([t[k] for k in ERRORS])
    k=(t['desi_teff']>=4200)&(t['desi_teff']<5000)
    positive=(t['label']==1)&k; wide=base & (positive | (t['label']==0))
    fields=np.unique(t['field_index'][base]).tolist(); records=[]
    for j in range(len(fields)):
        trainfields,calfields,field=field_roles(fields,j)
        train=np.flatnonzero(wide & np.isin(t['field_index'],trainfields))
        cal=np.flatnonzero(base & np.isin(t['field_index'],calfields))
        test=np.flatnonzero(base & (t['field_index']==field))
        for a,b in ((train,cal),(train,test),(cal,test)): assert_disjoint(t,a,b)
        old=k[train]; ref=train[old]; nref=len(ref)
        assert (positive[train]).sum()>=20 and (t['label'][ref]==0).sum()>=50
        realized=band_realizations(bands[train],errors[train],16,20261102+field)
        prep={d:preprocessing(bands[ref],errors[ref],d) for d in (2,3)}
        saved={'rows':test,'calibration_rows':cal,'identity_group':t['identity_group'][test]}
        rec={'test_field':field,'calibration_fields':calfields,'training_fields':trainfields,
             'n_reference':nref,'train_K_giants':int(positive[train].sum()),
             'train_K_dwarfs':int(((t['label'][ref]==0)).sum()),
             'train_all_dwarfs':int((t['label'][train]==0).sum()),'models':{}}
        for name,dim,subset in [('B-K',3,old),('A-wide',2,np.ones(len(train),bool)),('B-wide',3,np.ones(len(train),bool))]:
            model=fit_controlled(prep[dim],realized[subset],t['label'][train][subset],nref)
            color,_=colors_and_covariance(bands[train][subset],errors[train][subset],dim)
            # Shared K-domain is retained in the fitted object. Own domain is diagnostic.
            lower,upper=color.min(axis=0),color.max(axis=0)
            path=OUT/f'{field}_{name}.joblib';joblib.dump(model,path)
            modelhash=hashlib.sha256(path.read_bytes()).hexdigest()
            cs,_,co=predict(model,bands,errors,cal,cfg,20262000+field)
            ts,_,to=predict(model,bands,errors,test,cfg,20263000+field)
            saved['score_'+name]=ts;saved['cal_score_'+name]=cs;saved['outside_'+name]=to
            item={'model_sha256_before_calibration':modelhash,'total_weight':model['total_weight'],
                  'class_weight_totals':model['class_weight_totals'],'domains':{}}
            for domain in ('common_K','own'):
                if domain=='common_K': c_out,t_out=co,to
                else:
                    cc,_=colors_and_covariance(bands[cal],errors[cal],dim)
                    tc,_=colors_and_covariance(bands[test],errors[test],dim)
                    c_out=np.any((cc<lower)|(cc>upper),axis=1);t_out=np.any((tc<lower)|(tc>upper),axis=1)
                saved['outside_'+domain+'_'+name]=t_out
                rule={'0.5':{'threshold':.5}}
                for a in (.01,.001):rule[str(a)]=threshold_at_fpr(np.where(c_out,-1.,cs)[t['label'][cal]==0],a)
                item['domains'][domain]={'thresholds':rule,'test':{key:selection_summary(t,test,ts,t_out,masks['common5'],v['threshold'],True) for key,v in rule.items()}}
            assert hashlib.sha256(path.read_bytes()).hexdigest()==modelhash
            item['model_sha256_after_test']=modelhash;rec['models'][name]=item
        del realized
        np.savez_compressed(OUT/f'{field}_scores.npz',**saved)
        records.append(rec)
        (OUT/'progress.json').write_text(json.dumps({'completed_fields':len(records),'latest':field}))
        print(json.dumps({'field':field,'train':len(train),'cal':len(cal),'test':len(test)}),flush=True)
    sources=[Path(__file__),Path('src/photdis/giant_wide.py'),Path('docs/giant_wide_protocol_v1.md'),Path('data/giant_information_v1/labels.npz')]
    result={'folds':records,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'scope':'existing-field development validation; not untouched external test; scores are not calibrated probabilities'}
    (OUT/'results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')

if __name__=='__main__':main()

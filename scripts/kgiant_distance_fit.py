#!/usr/bin/env python3
"""Fixed empirical teacher mixtures, whole-field isolation and paired A/B scores."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
from photdis.stage1 import require_compute
from photdis.kgiant_distance import predict_mixture, evaluate_mixture
from kgiant_distance_inventory import OUT, sha, write_json


def main():
    require_compute()
    if (OUT/'predictions.json').exists(): raise FileExistsError(OUT/'predictions.json')
    with np.load(OUT/'population.npz') as f:q={k:f[k] for k in f.files}
    bands=np.column_stack([q[c] for c in ('g0','r0','z0','w10')])
    errors=np.column_stack([q[c] for c in ('sigma_g','sigma_r','sigma_z','sigma_w1')])
    assert np.all(np.isfinite(bands[q['w1_qualified']]))
    records=[];folds=[];outdir=OUT/'mixtures';outdir.mkdir(exist_ok=True)
    for teacher,widthkey in [('spdist','spdist_sigma_mu'),('specdis','specdis_sigma_mu'),('specdis_linear','specdis_sigma_mu_linear')]:
        kind=teacher.split('_')[0]
        eligible=q['w1_qualified'] & q[kind+'_valid'] & np.isfinite(q[widthkey]) & (q[widthkey]>0)
        mu=np.full(len(eligible),np.nan);mu[eligible]=10+5*np.log10(q[kind+'_dist_kpc'][eligible])
        for field in np.unique(q['field_index'][eligible]):
            test=np.flatnonzero(eligible & (q['field_index']==field))
            train=np.flatnonzero(eligible & (q['field_index']!=field) & (q['cluster_group']==0))
            train=train[~np.isin(q['identity_group'][train],q['identity_group'][test])]
            rec={'teacher':teacher,'field':int(field),'n_train':len(train),'n_test':len(test),
                 'n_train_fields':len(np.unique(q['field_index'][train]))}
            assert not set(q['identity_group'][train])&set(q['identity_group'][test])
            assert not set(q['source_id'][train])&set(q['source_id'][test])
            if len(train)<20 or rec['n_train_fields']<2:
                rec['status']='insufficient_training';folds.append(rec);continue
            rec['status']='scored';rec['train_source_sha256']=__import__('hashlib').sha256(q['source_id'][train].astype('<i8').tobytes()).hexdigest()
            folds.append(rec)
            for model,dimension in [('prior',0),('A_grz',2),('B_grzW1',3)]:
                mixtures=[]
                for i in test:
                    mixture=predict_mixture(bands[train],errors[train],mu[train],q[widthkey][train],
                                            bands[i],errors[i],dimension=dimension)
                    evaluated=evaluate_mixture(mixture,mu[i],q[widthkey][i])
                    records.append({'row':int(i),'source_id':int(q['source_id'][i]),'field':int(field),
                                    'teacher':teacher,'model':model,'selected':bool(q['selected'][i]),
                                    'reference_mu':float(mu[i]),'reference_sigma_mu':float(q[widthkey][i]),
                                    **evaluated})
                    mixtures.append(mixture)
                np.savez_compressed(outdir/f'{teacher}_{int(field)}_{model}.npz',
                                    rows=test,train_rows=train,train_source_id=q['source_id'][train],
                                    **{key:np.stack([m[key] for m in mixtures]) for key in ('means','sigmas','weights')})
            print(json.dumps(rec),flush=True)
    # In each teacher/field all model rows and actual training mixtures are identical.
    for fold in folds:
        if fold['status']!='scored':continue
        subsets=[[r['source_id'] for r in records if r['teacher']==fold['teacher'] and r['field']==fold['field'] and r['model']==m] for m in ('prior','A_grz','B_grzW1')]
        assert subsets[0]==subsets[1]==subsets[2]
    write_json(OUT/'predictions.json',{'folds':folds,'rows':records,
        'source_sha256':{str(p):sha(p) for p in [OUT/'population.npz',Path(__file__),Path('src/photdis/kgiant_distance.py'),Path('docs/kgiant_distance_protocol_v1.md')]},
        'parameters':{'color_bandwidth_mag':.12,'Mg_smoothing_mag':.15,'min_train':20,'min_train_fields':2},
        'scope':'existing-field development validation, teacher-scale error-broadened mixtures; no absolute calibration'})


if __name__=='__main__':main()

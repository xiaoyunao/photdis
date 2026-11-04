#!/usr/bin/env python3
"""Persist reproducible class weights and verify frozen expert/model hashes."""
from baseline_v1_3_run import *


def main():
    require_compute();t=load(OUT/'references.npz');checks={};weights={}
    for name in MODELS:
        ids=load(OUT/f'{name}_identities.npz');rows=ids['train_rows'];y=t['class_label'][rows];w=balanced_weights(y)
        checks[name+':identity']=np.array_equal(ids['train_identity'],t['identity'][rows])
        checks[name+':balanced_mass']=np.allclose([w[y==k].sum() for k in range(3)],len(y)/3)
        fit=json.load(open(V12/f'{name}_fit.json'))
        checks[name+':frozen_expert']=sha(EXPERT/f'{name}.joblib')==fit['expert_sha256']
        checks[name+':frozen_old_forest']=sha(V12/f'{name}_forest.joblib')==fit['forest']['model_sha256']
        path=OUT/f'{name}_weights.npz';np.savez_compressed(path,identity=ids['train_identity'],weight=w)
        weights[name]=dict(sha256=sha(path),class_counts=np.bincount(y,minlength=3),weight_sums=[w[y==k].sum() for k in range(3)])
    assert all(checks.values())
    write_json(OUT/'audit.json',dict(checks=checks,all_passed=True,n_checks=len(checks),weights=weights,
        result_sha256=sha(OUT/'result.json'),config_sha256=sha(ROOT/'configs/baseline_v1_2.json'),
        code_sha256=sha(ROOT/'scripts/baseline_v1_3_run.py')))
    print('Weight/frozen-model audit:',len(checks),'passed',flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Verify real frozen data, probability outputs and separation after production."""
import csv
import json
import numpy as np
import joblib
from v2_3_prepare import ROOT,DEST,TABLES
from v2_1_near import load,sha,save_json
from photdis.stage1 import require_compute
from photdis.probability_v2_1 import near_mass
from photdis.selection_v2_2 import canonical_probability


def main():
    require_compute();checks={}
    def check(name,result):
        checks[name]=bool(result)
        assert result,name
    freeze=json.loads((DEST/'models_frozen.json').read_text());tr=load(DEST/'training.npz');old=load(ROOT/'data/v2_1/frozen_training_audit.npz')
    for path,digest in freeze['inputs'].items():check('frozen:'+path,sha(ROOT/path)==digest)
    for name,h in freeze['models'].items():check('model:'+name,sha(DEST/(name+'.joblib'))==h)
    candidate=joblib.load(DEST/'candidate.joblib')
    check('candidate_converged',candidate.converged)
    check('covariances_positive',np.all(np.linalg.eigvalsh(candidate.covariances)>0))
    check('mixture_normalized',np.isclose(candidate.weights.sum(),1) and np.all(candidate.weights>0))
    check('auxiliary_class_normalized',np.allclose(candidate.class_frequency.sum(axis=1),1))
    check('likelihood_nondecreasing',np.min(np.diff(candidate.history))>-1e-5)
    check('strict_training',np.all(tr['TYPE']=='PSF'))
    check('unlabeled_support_retained',np.sum(tr['class_label']<0)==43914)
    check('finite_reference_errors',np.all(np.isfinite(tr['sigma_mu'])&(tr['sigma_mu']>0)))
    from photdis.reference_v1_4 import unique_join
    j=unique_join(tr['identity'],old['identity']);check('original_training_owners',(j>=0).all() and np.array_equal(tr['ls_id'],old['ls_id'][j]))
    for k in ['g0','r0','z0','sigma_g','sigma_r','sigma_z','Mg','TYPE']:check('unchanged_training:'+k,np.array_equal(tr[k],old[k][j],equal_nan=True) if tr[k].dtype.kind=='f' else np.array_equal(tr[k],old[k][j]))
    allids={}
    for role,source in [('development',ROOT/'data/v2_1/development.npz'),('reserved',DEST/'reserved_inputs.npz'),('application',ROOT/'data/v2_2/application_inputs.npz')]:
        t=load(source);allids[role]=t['identity'];check(role+':PSF',np.all(t['TYPE']=='PSF'))
        if role!='application':check(role+':no_fit_overlap',not np.intersect1d(t['identity'],tr['identity']).size)
        for name in ('control','candidate'):
            p=load(DEST/f'{role}_{name}.npz');grid=load(DEST/(name+'_grid.npz'))['edges']
            check(role+name+':identities',np.array_equal(t['identity'],p['identity']))
            check(role+name+':quantiles',np.all(np.isfinite(p['mu'])) and np.all(np.diff(p['mu'],axis=1)>=0))
            check(role+name+':q',np.array_equal(canonical_probability(p['qnear']),p['qnear']))
            count=0
            for path in sorted((DEST/'chunks').glob(f'{role}_{name}_*.npz')):
                q=load(path);r=q['rows'];key=path.stem
                check(key+':hash',sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256'])
                check(key+':PSF',np.all(q['TYPE']=='PSF'))
                check(key+':pdf',np.isfinite(q['pdf']).all() and np.all(q['pdf']>=0) and np.allclose(q['pdf'].sum(axis=1),1,atol=2e-6))
                check(key+':ids',np.array_equal(q['identity'],t['identity'][r]))
                check(key+':near_mass',np.max(abs(near_mass(q['pdf'],grid,t['g0'][r])-q['qnear']))<(1e-2 if name=='candidate' else 2e-6))
                if name=='candidate':check(key+':escaped_mass',np.max(q['escaped_mass'])<=1e-6)
                count+=len(r)
            check(role+name+':complete_rows',count==len(t['identity']))
    check('reserved_development_disjoint',not np.intersect1d(allids['reserved'],allids['development']).size)
    check('reserved_application_disjoint',not np.intersect1d(allids['reserved'],allids['application']).size)
    opened=json.loads((DEST/'reserved_opened.json').read_text());check('reserved_models_frozen_first',opened['models']==freeze['models'])
    for role in ('development','reserved'):
        with open(TABLES/f'{role}_pit.csv') as f:rows=list(csv.DictReader(f))
        with open(TABLES/f'{role}_gaia.csv') as f:ga=list(csv.DictReader(f))
        key=lambda r:tuple(r[k] for k in ('role','model','group','selection'))
        counts={}
        for r in rows:counts[key(r)]=counts.get(key(r),0)+int(r['n'])
        for i,r in enumerate(ga):check(role+':PITdenominator:'+str(i),counts[key(r)]==int(r['n']))
    save_json(DEST/'validation.json',dict(actual_checks=len(checks),all_passed=all(checks.values()),checks=checks))
    print('VALIDATION',len(checks),flush=True)


if __name__=='__main__':main()

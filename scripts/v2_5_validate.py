#!/usr/bin/env python3
"""Actual-source, role, normalization and fresh-catalog validation."""
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
D=ROOT/'data/v2_5';T=D/'tables'


def main():
    require_compute();checks={}
    def check(k,v):checks[k]=bool(v);assert v,k
    for stage in ('diagnostic','population'):
        for path,h in json.loads((D/(stage+'_freeze.json')).read_text()).items():check('freeze:'+path,sha(ROOT/path)==h)
    freeze=json.loads((ROOT/'data/v2_3/models_frozen.json').read_text())
    for name,h in freeze['models'].items():check('original_model:'+name,sha(ROOT/'data/v2_3'/(name+'.joblib'))==h)
    c=load(D/'cal_templates.npz');t=load(D/'test_templates.npz');a=load(D/'catalogs.npz');fit=load(ROOT/'data/v2_3/training.npz')['identity']
    check('cal_test_disjoint',not np.intersect1d(c['identity'],t['identity']).size)
    for name,q in [('cal',c),('test',t),('catalog',a)]:
        check(name+':PSF',(q['TYPE']=='PSF').all());check(name+':no_fit_overlap',not np.intersect1d(q['identity'],fit).size)
    check('missing_labels_retained',np.any(~c['label_available']));check('label_marginal_normalization',np.allclose(c['member'].sum(axis=1),1))
    check('192_fresh_catalogs',len(np.unique(a['catalog']))==192)
    for i in np.unique(a['catalog']):
        u=a['catalog']==i;check(f'catalog:{i}:size',u.sum()==128)
        selection=(a['mags'][u,1]>=14)&(a['mags'][u,1]<20)&(a['errors'][u].max(axis=1)<.1)
        check(f'catalog:{i}:selection',np.all(selection==(a['regime'][u]==0)))
    # Same-source replicas may recur; whole independent catalog random draws are
    # distinct, rather than reusing one fixed noisy table as observed realizations.
    fingerprints=[sha_bytes(a['mags'][a['catalog']==i]) for i in np.unique(a['catalog'])]
    check('unique_catalog_noise',len(set(fingerprints))==192)
    for path in sorted((D/'qrf_chunks').glob('*.npz')):
        check('QRFhash:'+path.name,sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256'])
        q=load(path);r=q['rows'];check('QRFids:'+path.name,np.array_equal(q['identity'],a['identity'][r]))
        check('QRFmass:'+path.name,np.allclose(q['joint_mass'][:,:4].sum(axis=(1,2)),1,atol=2e-6))
    for regime in range(2):
        q=load(D/f'posterior_regime{regime}.npz');check(f'posterior{regime}:norm',np.allclose(q['weights'].sum(axis=1),1))
        check(f'component{regime}:norm',np.allclose(q['selected'].sum(axis=1),1));check(f'response{regime}:alpha',np.all(q['response'].sum(axis=1)<=1+1e-10))
        check(f'posterior{regime}:finite',np.isfinite(q['weights']).all())
    with open(T/'recovery_distance.csv') as f:rows=list(csv.DictReader(f))
    summaries=[]
    from photdis.selection_v2_2 import wilson
    for design in sorted({r['design'] for r in rows}):
        for regime in ('0','1'):
            rr=[r for r in rows if r['design']==design and r['regime']==regime]
            for method in ('point','pdf','population'):
                errors=[]
                for i in sorted({r['catalog'] for r in rr}):
                    q=[r for r in rr if r['catalog']==i];check('five_bins:'+i,len(q)==5)
                    if method=='population':check('population_mass:'+i,np.isclose(sum(float(r[method]) for r in q),1,atol=1e-7))
                    errors.append(sum(abs(float(r[method])-float(r['truth'])) for r in q) + (abs(1-sum(float(r[method]) for r in q)) if method in ('point','pdf') else 0))
                summaries.append(dict(design=design,regime=regime,method=method,catalogs=len(errors),mean_L1=float(np.mean(errors)),sd_L1=float(np.std(errors,ddof=1))))
            for b in range(5):
                q=[r for r in rr if int(r['bin'])==b];n=len(q)
                for kind,target,lo,hi in [('parameter','truth','low','high'),('predictive','realized','predictive_low','predictive_high')]:
                    success=sum(float(r[lo])<=float(r[target])<=float(r[hi]) for r in q);bounds=wilson(success,n)
                    summaries.append(dict(design=design,regime=regime,method=kind,bin=b,catalogs=n,coverage=success/n,wilson_low=bounds[0],wilson_high=bounds[1]))
    csv_rows(T/'recovery_summary.csv',summaries)
    tail=[]
    for i in sorted({r['catalog'] for r in rows}):
        q=[r for r in rows if r['catalog']==i]
        tail.append(dict(catalog=i,design=q[0]['design'],regime=q[0]['regime'],truth=0,point=1-sum(float(r['point']) for r in q),pdf=1-sum(float(r['pdf']) for r in q),population=0))
    csv_rows(T/'recovery_beyond50.csv',tail)
    meta=json.loads((ROOT/'data/v2_4/reservation.json').read_text());check('reservation_still_unopened',not meta['reference_outcomes_opened'] and not meta['predictions_generated'])
    save_json(D/'validation.json',dict(actual_checks=len(checks),all_passed=all(checks.values()),checks=checks,job=os.getenv('SLURM_JOB_ID')))
    print('VALIDATION',len(checks),flush=True)


def sha_bytes(a):
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()


if __name__=='__main__':main()

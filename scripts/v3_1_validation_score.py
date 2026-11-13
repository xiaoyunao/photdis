#!/usr/bin/env python3
"""Same-identity final scores and actual stored-level numerical tail checks."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.freeze import verify_freeze
from photdis.v3_1.validation_scores import score_pdf,compare_score_levels

_EVIDENCE=_COHORT=_OUT=_INPUTS=_COMMON=None
REFERENCE_VIEWS = ('nominal','response12','corrected_point','corrected_bootstrap')
TARGET_VIEWS = ('zero','point','bootstrap_evidence','bootstrap_cut')


def levels(family, identity):
    root = ROOT/'data/v3_1'
    folder = {'reference':'validation_reference_refinement','old_target':'validation_old_target',
              'candidate':'validation_target/candidate'}[family]
    obj = root/folder/_COHORT/f'object_{identity}'
    record = json.loads((obj/'complete.json').read_text())
    attempts = record['attempts']; selected = attempts[-1]
    def read(attempt):
        meta = json.loads((obj/attempt['marker']).read_text())
        data = load(obj/f'step{attempt["step"]:g}.npz')
        return data,meta
    current = read(selected)
    if len(attempts)>1:
        previous = read(attempts[-2])
    else:
        assert family == 'reference'
        path = root/f'validation_reference_predictions/{_COHORT}/object_{identity}'
        previous = load(path.with_suffix('.npz')),json.loads(path.with_suffix('.json').read_text())
    return previous,current,record


def one(row):
    identity = int(_EVIDENCE['identity'][row]); path = _OUT/f'object_{identity}.json'
    if path.exists():
        record = json.loads(path.read_text()); assert record['inputs'] == _INPUTS
        return record
    evidence = {k:v[row] for k,v in _EVIDENCE.items()}; views = {}; checks = []
    common = _COMMON.get(int(evidence['scale_system']))
    for family in ('reference','old_target','candidate'):
        previous,current,metadata = levels(family,identity)
        names = REFERENCE_VIEWS if family == 'reference' else TARGET_VIEWS if family == 'candidate' else ('old_target',)
        for name in names:
            results = []
            for data,meta in (previous,current):
                value = meta if family == 'old_target' else meta['results'][name]
                lp = data['logpdf' if family == 'old_target' else name+'_logpdf']
                results.append(score_pdf(data['grid'],lp,value['mu'],value['class_distance'],evidence,common_scale=common))
            key = family+'/'+name; comparison = compare_score_levels(*results)
            if family == 'old_target':
                prior_pass = current[1]['passed']; inherited = True
            else:
                key_name = 'view' if family == 'candidate' else 'variant'
                prior_pass = next(c['passed'] for c in current[1]['checks'] if c[key_name] == name)
                inherited = current[1]['results'][name].get('inherited_ensemble_accepted',True)
            views[key] = results[1]
            checks.append(dict(view=key,previous_numerical_passed=prior_pass,inherited_ensemble_accepted=inherited,
                **comparison,accepted=bool(prior_pass and inherited and comparison['passed'])))
    record = dict(inputs=_INPUTS,identity=identity,TYPE='PSF',role=str(evidence['role']),cell=int(evidence['cell']),
        class_label=int(evidence['class_label']),scale_system=int(evidence['scale_system']),views=views,checks=checks,
        passed=all(r['accepted'] for r in checks),job=os.getenv('SLURM_JOB_ID'))
    save_json(path,record); return record


def main():
    global _EVIDENCE,_COHORT,_OUT,_INPUTS,_COMMON
    require_compute(); p=argparse.ArgumentParser(); p.add_argument('--cohort',choices=['field','common'],required=True)
    p.add_argument('--workers',type=int,default=8); a=p.parse_args(); assert 1<=a.workers<=32
    _COHORT=a.cohort; root=ROOT/'data/v3_1/final_validation'; freeze=root/'freeze.json'
    digest=verify_freeze(ROOT,freeze); source=root/a.cohort; done=json.loads((source/'complete.json').read_text())
    assert done['inputs']['freeze_sha256']==digest and done['final_outcomes_opened']
    for name,checksum in done['outputs'].items():assert sha(source/name)==checksum
    _INPUTS={str(p.relative_to(ROOT)):sha(p) for p in [freeze,source/'complete.json',source/'evidence.npz',
        ROOT/'scripts/v3_1_validation_score.py',ROOT/'src/photdis/v3_1/validation_scores.py']}
    _EVIDENCE=load(source/'evidence.npz'); assert (_EVIDENCE['TYPE']=='PSF').all(); _COMMON={}
    if a.cohort=='common':
        for i,scale in enumerate(json.loads((source/'scales.json').read_text())):
            if not (_EVIDENCE['scale_system']==i).any():continue
            assert scale['system']=='NGC_1261' and scale['method']=='hst_kinematic'
            r=scale['reference']; assert r['minus_kpc']==r['plus_kpc']
            _COMMON[i]=(r['distance_kpc'],r['plus_kpc'])
    _OUT=root/'scores'/a.cohort; _OUT.mkdir(parents=True,exist_ok=True); inputs=_OUT/'inputs.json'
    if inputs.exists():assert json.loads(inputs.read_text())==_INPUTS
    else:save_json(inputs,_INPUTS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        records=list(pool.map(one,range(len(_EVIDENCE['identity']))))
    rows=[dict(identity=r['identity'],TYPE='PSF',role=r['role'],view=c['view'],accepted=c['accepted'],
        channels_checked=c['channels_checked'],score_numerical_passed=c['passed'],
        previous_numerical_passed=c['previous_numerical_passed'],inherited_ensemble_accepted=c['inherited_ensemble_accepted'])
        for r in records for c in r['checks']]
    csv_rows(_OUT/'numerical.csv',rows)
    save_json(_OUT/'complete.json',dict(inputs=_INPUTS,objects={f'object_{r["identity"]}.json':sha(_OUT/f'object_{r["identity"]}.json') for r in records},
        outputs={'numerical.csv':sha(_OUT/'numerical.csv')},n=len(records),cohort=a.cohort,all_PSF=True,
        passed=all(r['passed'] for r in records),failed_views=sum(not r['accepted'] for r in rows),job=os.getenv('SLURM_JOB_ID'),
        scope='Nine same-identity views with actual previous-resolution proper-score/PIT checks. No row dropped; shared-scale/sky aggregate report still required.'))


if __name__=='__main__':main()

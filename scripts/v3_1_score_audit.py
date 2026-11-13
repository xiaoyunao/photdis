#!/usr/bin/env python3
"""Audit every stored external score, denominator and numerical failure channel."""
import collections
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute


def main():
    require_compute();root=ROOT/'data/v3_1/final_validation';out=root/'score_audit';out.mkdir(exist_ok=True)
    inputs={};checks=collections.Counter();views=[];failures=[];channels=[];inventories=[]
    expected_views={'reference/'+v for v in ['nominal','response12','corrected_point','corrected_bootstrap']}
    expected_views |= {'candidate/'+v for v in ['zero','point','bootstrap_evidence','bootstrap_cut']} | {'old_target/old_target'}
    def verify(condition,name):
        checks[name]+=1
        if not condition:raise AssertionError(name)
    freeze=root/'freeze.json';digest=sha(freeze);inputs[str(freeze.relative_to(ROOT))]=digest
    for cohort,n in [('field',526),('common',3032)]:
        folder=root/'scores'/cohort;marker=folder/'complete.json';done=json.loads(marker.read_text())
        evidence_marker=root/cohort/'complete.json';ed=json.loads(evidence_marker.read_text());evidence_path=root/cohort/'evidence.npz'
        for path in [marker,evidence_marker,evidence_path,folder/'numerical.csv']:inputs[str(path.relative_to(ROOT))]=sha(path)
        verify(done['n']==ed['n']==n and done['all_PSF'] and ed['all_PSF'],'cohort_size_PSF')
        verify(ed['inputs']['freeze_sha256']==digest and done['inputs'][str(freeze.relative_to(ROOT))]==digest,'immutable_freeze_link')
        verify(sha(evidence_path)==ed['outputs']['evidence.npz'],'evidence_hash')
        verify(sha(folder/'numerical.csv')==done['outputs']['numerical.csv'],'numerical_table_hash')
        evidence=load(evidence_path);verify(len(evidence['identity'])==n and (evidence['TYPE']=='PSF').all(),'evidence_rows_PSF')
        expected={f'object_{int(i)}.json' for i in evidence['identity']};verify(set(done['objects'])==expected,'complete_identity_set')
        with (folder/'numerical.csv').open() as stream:table=list(csv.DictReader(stream))
        index={(int(r['identity']),r['view']):r for r in table};verify(len(index)==len(table)==9*n,'unique_nine_view_table')
        byview=collections.defaultdict(collections.Counter);bychannel={};failed_ids=set()
        for row,identity in enumerate(evidence['identity']):
            path=folder/f'object_{identity}.json';verify(sha(path)==done['objects'][path.name],'object_hash');r=json.loads(path.read_text())
            verify(r['inputs']==done['inputs'],'object_input_identity')
            verify(r['identity']==identity and r['TYPE']=='PSF' and r['role']==str(evidence['role'][row]) and r['cell']==int(evidence['cell'][row]),'object_role_PSF')
            verify(len(r['views'])==len(r['checks'])==9 and {c['view'] for c in r['checks']}==set(r['views'])==expected_views,'object_nine_views')
            eligible=set()
            for channel,valid in [('astronn',evidence['distance_available'][row]),('gaia',evidence['parallax_eligible'][row]),
                ('common_shared_scale',evidence['scale_system'][row]>=0),('class',evidence['class_label'][row]>=0),
                ('mws_class',evidence['mws_class_label'][row]>=0),('specdis',evidence['specdis_valid'][row]),('spdist',evidence['spdist_valid'][row])]:
                if valid:eligible.add(channel)
            for check in r['checks']:
                view=check['view'];value=r['views'][view];p=np.asarray(value['class_distance']);q=np.asarray(value['mu'])
                verify(p.shape==(3,6) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-8,'class_distance_mass')
                verify(abs(value['qnear']-p[:,:3].sum())<1e-10 and np.allclose(value['class_probability'],p.sum(axis=1),atol=1e-10,rtol=0),'near_class_marginals')
                verify(q.shape==(5,) and np.isfinite(q).all() and (np.diff(q)>=0).all() and abs(value['width68']-(q[3]-q[1]))<1e-10 and abs(value['width95']-(q[4]-q[0]))<1e-10,'ordered_quantiles_widths')
                cs=check['checks'];verify(check['channels_checked']==len(cs) and {c['channel'] for c in cs}==eligible,'qualified_channel_denominator')
                verify(len({c['channel'] for c in cs})==len(cs),'unique_channels')
                verify(check['passed']==all(c['passed'] for c in cs),'score_gate_conjunction')
                verify(check['accepted']==bool(check['passed'] and check['previous_numerical_passed'] and check['inherited_ensemble_accepted']),'inherited_gate_conjunction')
                exported=index[(int(identity),view)]
                for target,key in [('accepted','accepted'),('score_numerical_passed','passed'),('previous_numerical_passed','previous_numerical_passed'),('inherited_ensemble_accepted','inherited_ensemble_accepted')]:
                    verify(exported[target]==str(check[key]),'exported_flag_matches_record')
                counts=byview[view];counts['n']+=1;counts['accepted']+=check['accepted'];counts['score_numerical_passed']+=check['passed']
                counts['inherited_failed']+=not check['inherited_ensemble_accepted'];counts['previous_failed']+=not check['previous_numerical_passed']
                for c in cs:
                    delta=c['logscore_delta'];pit=c.get('pit_delta',0.)
                    finite=delta is not None and pit is not None and np.isfinite([delta,pit]).all()
                    gate=finite and delta>=0 and pit>=0 and delta<.02 and pit<.003 and not c.get('nonfinite_class_score',False)
                    verify(c['passed']==bool(gate),'stored_level_threshold')
                    key=(view,c['channel']);stats=bychannel.setdefault(key,dict(n=0,failed=0,nonfinite=0,max_logscore_delta=0.,max_PIT_delta=0.))
                    stats['n']+=1;stats['failed']+=not c['passed'];stats['nonfinite']+=not finite
                    if finite:stats['max_logscore_delta']=max(stats['max_logscore_delta'],delta);stats['max_PIT_delta']=max(stats['max_PIT_delta'],pit)
                    if not c['passed']:
                        failed_ids.add(int(identity));failures.append(dict(cohort=cohort,identity=int(identity),TYPE='PSF',view=view,channel=c['channel'],
                            logscore_delta=delta,PIT_delta=pit,nonfinite_class_score=c.get('nonfinite_class_score',False),inherited_ensemble_accepted=check['inherited_ensemble_accepted']))
            verify(r['passed']==all(c['accepted'] for c in r['checks']),'object_gate_conjunction')
        verify(sum(v['n']-v['accepted'] for v in byview.values())==done['failed_views'],'complete_failed_view_count')
        verify(done['passed']==all(v['accepted']==n for v in byview.values()),'complete_acceptance_flag')
        views.extend(dict(cohort=cohort,view=view,**counts) for view,counts in sorted(byview.items()))
        channels.extend(dict(cohort=cohort,view=view,channel=channel,**stats) for (view,channel),stats in sorted(bychannel.items()))
        inventories.append(dict(cohort=cohort,n=n,view_rows=9*n,objects_with_score_failure=len(failed_ids)))
    inputs[str((ROOT/'scripts/v3_1_score_audit.py').relative_to(ROOT))]=sha(ROOT/'scripts/v3_1_score_audit.py')
    csv_rows(out/'views.csv',views);csv_rows(out/'channels.csv',channels);csv_rows(out/'failures.csv',failures)
    save_json(out/'checks.json',dict(checks=dict(checks),actual_checks=sum(checks.values())))
    save_json(out/'complete.json',dict(inputs=inputs,outputs={name:sha(out/name) for name in ['views.csv','channels.csv','failures.csv','checks.json']},
        inventory=inventories,actual_checks=sum(checks.values()),artifact_audit_passed=True,scientific_acceptance=False,
        job=os.getenv('SLURM_JOB_ID'),scope='Audit every stored score/identity/channel/flag and exported numerical row. Does not recompute PDFs, change frozen outcomes, erase numerical failures or establish scientific calibration.'))


if __name__=='__main__':main()

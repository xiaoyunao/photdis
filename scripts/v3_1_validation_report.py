#!/usr/bin/env python3
"""Post-freeze external report with paired sky blocks and one shared HST scale."""
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.freeze import verify_freeze
from photdis.v3_1.validation_report import (MAIN_VIEW,external_strata,metric_arrays,
    field_statistic,paired_field_statistic,common_distance_draws,common_scale_statistics)


def read_cohort(root,cohort,inputs):
    source=root/'final_validation'; score=source/'scores'/cohort
    paths=[score/'complete.json',source/cohort/'complete.json',root/f'validation_roles/{cohort}_photometry.npz']
    for path in paths:inputs[str(path.relative_to(ROOT))]=sha(path)
    done=json.loads(paths[0].read_text()); evidence_done=json.loads(paths[1].read_text())
    assert done['all_PSF'] and done['cohort']==cohort and evidence_done['cohort']==cohort
    for name,digest in evidence_done['outputs'].items():assert sha(source/cohort/name)==digest
    evidence=load(source/cohort/'evidence.npz'); photo=load(paths[2]); np.testing.assert_array_equal(photo['identity'],evidence['identity'])
    am=root/'validation_admission/complete.json'; ad=json.loads(am.read_text()); ap=am.parent/(cohort+'.npz')
    assert sha(ap)==ad['outputs'][ap.name]; inputs[str(am.relative_to(ROOT))]=sha(am)
    admission=load(ap); np.testing.assert_array_equal(admission['identity'],photo['identity'])
    assert (admission['TYPE']=='PSF').all(); photo['candidate_admission']=admission['candidate']
    assert (photo['TYPE']=='PSF').all() and (evidence['TYPE']=='PSF').all()
    expected={f'object_{int(i)}.json' for i in photo['identity']}; assert set(done['objects'])==expected
    records=[]
    for i,identity in enumerate(photo['identity']):
        path=score/f'object_{int(identity)}.json'; assert sha(path)==done['objects'][path.name]
        r=json.loads(path.read_text()); assert r['identity']==identity and r['TYPE']=='PSF'
        assert r['inputs']==done['inputs'] and r['role']==str(photo['role'][i]) and r['cell']==int(photo['cell'][i])
        records.append(r)
    assert len(records)==done['n']==evidence_done['n']
    views=sorted(records[0]['views']); assert len(views)==9 and MAIN_VIEW in views
    assert all(set(r['views'])==set(views) for r in records)
    metrics={view:metric_arrays(records,view) for view in views}
    return photo,evidence,records,views,metrics


def field_report(photo,evidence,records,views,metrics,out):
    admit=photo['candidate_admission']; masks={'candidate/'+name:mask & admit for name,mask in external_strata(photo,evidence).items()}
    masks.update({'diagnostic_all_photometry/all':np.ones(len(admit),bool),'outside_candidate_domain/all':~admit})
    cells=photo['cell']; summaries=[]; paired=[]; availability=[]; highq=[]; confusion=[]
    main,main_ok=metrics[MAIN_VIEW]; selected=admit & (main['qnear']>.95)
    for role in ['new_sky_field','new_source_seen_sky_field']:
        role_mask=photo['role']==role
        for group,group_mask in masks.items():
            mask=role_mask & group_mask
            availability.append(dict(role=role,stratum=group,n=int(mask.sum()),cells=len(np.unique(cells[mask])),
                distance=int((mask & evidence['distance_available']).sum()),precise_distance=int((mask & evidence['distance_precise']).sum()),
                signed_Gaia=int((mask & evidence['parallax_eligible']).sum()),class_dwarf=int((mask & (evidence['class_label']==0)).sum()),
                class_giant=int((mask & (evidence['class_label']==1)).sum()),feh=int((mask & evidence['feh_valid']).sum()),
                SpecDis=int((mask & evidence['specdis_valid']).sum()),SPDist=int((mask & evidence['spdist_valid']).sum()),
                MWS_dwarf=int((mask & (evidence['mws_class_label']==0)).sum()),MWS_giant=int((mask & (evidence['mws_class_label']==1)).sum()),
                MWS_other=int((mask & (evidence['mws_class_label']==2)).sum()),MWS_identity_conflict=int((mask & evidence['mws_identity_conflict']).sum()),
                finalcandidate_numerical_failed=int((mask & ~main_ok).sum()),fixed_main_highq=int((mask & selected).sum())))
            for view in views:
                values,passed=metrics[view]
                for admission in ['all_objects','numerically_accepted']:
                    use=mask & (passed if admission=='numerically_accepted' else True)
                    for name,x in values.items():
                        qualifications=[('available',np.ones(len(mask),bool))]
                        if name.startswith('astronn/'):
                            qualifications.append(('precise',evidence['distance_precise']))
                        for qualification,qmask in qualifications:
                            m=use & qmask
                            summaries.append(dict(role=role,stratum=group,view=view,admission=admission,qualification=qualification,
                                metric=name,group_n=int(mask.sum()),group_numerical_failed=int((mask & ~passed).sum()),
                                **field_statistic(x[m],cells[m])))
                    if view==MAIN_VIEW:continue
                    use=mask & ((passed & main_ok) if admission=='numerically_accepted' else True)
                    for name in sorted(set(main) & set(values)):
                        qualifications=[('available',np.ones(len(mask),bool))]
                        if name.startswith('astronn/'):qualifications.append(('precise',evidence['distance_precise']))
                        for qualification,qmask in qualifications:
                            m=use & qmask
                            paired.append(dict(role=role,stratum=group,baseline=view,main=MAIN_VIEW,admission=admission,qualification=qualification,
                                metric=name,group_n=int(mask.sum()),paired_numerical_failed=int((mask & ~(passed & main_ok)).sum()),
                                **paired_field_statistic(main[name][m],values[name][m],cells[m])))
        fixed=role_mask & selected
        for view in views:
            values,passed=metrics[view]
            channels=[('astronn','available',evidence['distance_available'],evidence['distance_kpc']),
                ('astronn','precise',evidence['distance_precise'],evidence['distance_kpc'])]
            channels += [(name,'qualified',evidence[name+'_valid'],evidence[name+'_distance']) for name in ['specdis','spdist']]
            for channel,qualification,qmask,distance in channels:
                qualified=fixed & qmask; n=int(fixed.sum()); nr=int(qualified.sum())
                highq.append(dict(role=role,view=view,selection_view=MAIN_VIEW,n=n,reference_channel=channel,reference_qualification=qualification,
                    model_expected_far=float((1-values['qnear'][fixed]).sum()),
                    model_expected_leakage=float((1-values['qnear'][fixed]).mean()) if n else None,
                    reference_n=nr,reference_center_far=int((qualified & (distance>=10)).sum()),
                    reference_center_far_fraction=float((distance[qualified]>=10).mean()) if nr else None,
                    numerical_failed=int((fixed & ~passed).sum()),main_numerical_failed=int((fixed & ~main_ok).sum()),
                    scope='Reference-selected disagreement; not independently measured fullcandidate contamination.'))
            for channel,label_key,labels in [('class','class_label',[0,1]),('mws_class','mws_class_label',[0,1,2])]:
                for label in labels:
                    for predicted in [0,1,2]:
                        for admission in ['all_objects','numerically_accepted']:
                            use=role_mask & admit & (evidence[label_key]==label) & (passed if admission=='numerically_accepted' else True)
                            ids=np.flatnonzero(use)
                            count=sum(records[i]['views'][view][channel]['predicted']==predicted for i in ids)
                            confusion.append(dict(role=role,view=view,admission=admission,class_source=channel,reference_class=label,predicted_class=predicted,n=count,denominator=len(ids)))
    # Keep role files separate so exact sparse-cell tables remain inspectable small artifacts.
    outputs=[]
    for role in ['new_sky_field','new_source_seen_sky_field']:
        for label,rows in [('summary',summaries),('paired',paired),('availability',availability),('highq',highq),('confusion',confusion)]:
            chosen=[r for r in rows if r['role']==role]
            if label in ('summary','paired'):
                channels=sorted({r['metric'].split('/')[0] if '/' in r['metric'] else 'photometric' for r in chosen})
                for channel in channels:
                    for admission in ['all_objects','numerically_accepted']:
                        subset=[r for r in chosen if r['admission']==admission and
                            (r['metric'].split('/')[0] if '/' in r['metric'] else 'photometric')==channel]
                        name=f'{role}_{label}_{channel}_{admission}.csv'; csv_rows(out/name,subset); outputs.append(name)
            else:
                name=f'{role}_{label}.csv'; csv_rows(out/name,chosen); outputs.append(name)
    return outputs


def draw_interval(values):
    values=np.asarray(values)
    if not len(values) or np.isnan(values).all():return dict(lower95=None,median=None,upper95=None)
    assert np.isfinite(values).all()
    lo,median,hi=np.quantile(values,[.025,.5,.975])
    return dict(lower95=float(lo),median=float(median),upper95=float(hi))


def common_report(root,photo,evidence,records,views,metrics,out,inputs):
    path=root/'final_validation/common/scales.json'; inputs[str(path.relative_to(ROOT))]=sha(path)
    scales=json.loads(path.read_text()); main,main_ok=metrics[MAIN_VIEW]
    rows=[]; paired=[]; descriptive=[]; arrays={}; descriptors=[]
    for system in np.unique(evidence['scale_system']):
        assert system>=0; scale=scales[int(system)]; ref=scale['reference']
        assert scale['system']=='NGC_1261' and scale['method']=='hst_kinematic' and ref['minus_kpc']==ref['plus_kpc']
        distance=common_distance_draws(ref['distance_kpc'],ref['plus_kpc'])
        arrays[f'system_{system}_distance_kpc']=distance
        system_mask=evidence['scale_system']==system
        admit=photo['candidate_admission']
        for subset,subset_mask in [('candidate_members',system_mask & admit),('all_members_diagnostic',system_mask),
                ('outside_candidate_domain',system_mask & ~admit),('fixed_main_highq',system_mask & admit & (main['qnear']>.95))]:
            for admission in ['all_objects','numerically_accepted']:
                for view in views:
                    values,passed=metrics[view]; use=subset_mask & (passed if admission=='numerically_accepted' else True)
                    q=np.array([r['views'][view]['mu'] for r in records])[use]
                    for name,x in values.items():
                        good=use & ~np.isnan(x); finite=np.isfinite(x[good]).all()
                        descriptive.append(dict(system=scale['system'],subset=subset,view=view,admission=admission,metric=name,
                            group_n=int(subset_mask.sum()),n=int(good.sum()),independent_systems=1,
                            nonfinite=int((~np.isfinite(x[good])).sum()),mean=float(x[good].mean()) if good.any() and finite else None,
                            scope='Marginal per-object descriptive statistic; no independent-member uncertainty interval.'))
                    for factor in [.97,1.,1.03]:
                        stats=common_scale_statistics(q,distance*factor)
                        for metric,samples in stats.items():
                            key=f'draws_{len(descriptors):04d}'; arrays[key]=samples
                            descriptor=dict(key=key,system=scale['system'],subset=subset,admission=admission,view=view,metric=metric,method_factor=factor)
                            descriptors.append(descriptor)
                            rows.append(dict(**descriptor,n=int(use.sum()),group_n=int(subset_mask.sum()),independent_scales=1,
                                numerical_failed=int((subset_mask & ~passed).sum()),**draw_interval(samples)))
                    if view==MAIN_VIEW:continue
                    match=subset_mask & ((passed & main_ok) if admission=='numerically_accepted' else True)
                    mq=np.array([r['views'][MAIN_VIEW]['mu'] for r in records])[match]
                    bq=np.array([r['views'][view]['mu'] for r in records])[match]
                    for factor in [.97,1.,1.03]:
                        left=common_scale_statistics(mq,distance*factor); right=common_scale_statistics(bq,distance*factor)
                        for metric in left:
                            delta=left[metric]-right[metric]; key=f'draws_{len(descriptors):04d}'; arrays[key]=delta
                            descriptor=dict(key=key,system=scale['system'],subset=subset,admission=admission,main=MAIN_VIEW,baseline=view,
                                metric=metric,method_factor=factor)
                            descriptors.append(descriptor)
                            paired.append(dict(**descriptor,n=int(match.sum()),group_n=int(subset_mask.sum()),independent_scales=1,
                                paired_numerical_failed=int((subset_mask & ~(passed & main_ok)).sum()),**draw_interval(delta)))
    arrays['shared_scale_draw_id']=np.arange(4096,dtype='i8')
    atomic_npz(out/'common_scale_draws.npz',**arrays); save_json(out/'common_draw_columns.json',descriptors)
    csv_rows(out/'common_scale_summary.csv',rows); csv_rows(out/'common_scale_paired.csv',paired)
    csv_rows(out/'common_marginal_summary.csv',descriptive)
    return ['common_scale_draws.npz','common_draw_columns.json','common_scale_summary.csv','common_scale_paired.csv','common_marginal_summary.csv']


def main():
    require_compute(); root=ROOT/'data/v3_1'; freeze=root/'final_validation/freeze.json'; digest=verify_freeze(ROOT,freeze)
    out=root/'final_validation/report'; out.mkdir(exist_ok=True)
    inputs={str(path.relative_to(ROOT)):sha(path) for path in [freeze,ROOT/'scripts/v3_1_validation_report.py',
        ROOT/'src/photdis/v3_1/validation_report.py',ROOT/'docs/v3_1_final_validation_protocol.md']}
    outputs=[]; inventory=[]
    for cohort in ['field','common']:
        photo,evidence,records,views,metrics=read_cohort(root,cohort,inputs)
        assert json.loads((root/f'final_validation/{cohort}/complete.json').read_text())['inputs']['freeze_sha256']==digest
        inventory.append(dict(cohort=cohort,n=len(records),views=len(views),all_PSF=True,
            failed_objects=sum(not r['passed'] for r in records),failed_views=sum(not c['accepted'] for r in records for c in r['checks'])))
        if cohort=='field':outputs+=field_report(photo,evidence,records,views,metrics,out)
        else:outputs+=common_report(root,photo,evidence,records,views,metrics,out,inputs)
    save_json(out/'complete.json',dict(inputs=inputs,outputs={name:sha(out/name) for name in outputs},inventory=inventory,
        job=os.getenv('SLURM_JOB_ID'),all_PSF=True,all_frozen_identities_retained=True,
        scientific_acceptance=False,independent_fullcandidate_contamination_measured=False,
        scope='Pre-registered paired sky-block field and one shared HST scale report. Empty/flagged cells retained. Common draw intervals describe reference-scale sensitivity, not independently measured population coverage.'))


if __name__=='__main__':main()

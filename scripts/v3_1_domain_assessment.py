#!/usr/bin/env python3
"""Paired final-reference screens on the frozen observable domain cells."""
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from v3_1_validation_report import read_cohort
from photdis.stage1 import require_compute
from photdis.v3_1.freeze import verify_freeze
from photdis.v3_1.validation_report import MAIN_VIEW
from photdis.v3_1.use_domains import N_CELLS,source_distances
from photdis.v3_1.domain_assessment import BASELINE,assess_pair,summarize_cell


def field_cell_screens(photo,evidence,metrics,domain):
    np.testing.assert_array_equal(photo['identity'],domain['identity'])
    np.testing.assert_array_equal(photo['candidate_admission'],domain['candidate'])
    if not (photo['TYPE']=='PSF').all() or not (domain['TYPE']=='PSF').all():raise ValueError('Native PSF required')
    main,main_ok=metrics[MAIN_VIEW]; base,base_ok=metrics[BASELINE]; n=len(photo['identity'])
    admit=photo['candidate_admission']; groups={'all':admit,'fixed_main_highq':admit & (main['qnear']>.95)}
    for source,key,labels in [('apogee','class_label',[-1,0,1]),('mws','mws_class_label',[-1,0,1,2])]:
        for label in labels:groups[f'{source}_class_{label}']=admit & (evidence[key]==label)
    channels=[]
    for source,(_,valid,precise) in source_distances(evidence).items():
        for quality,qmask in [('qualified',valid),('precise',precise)]:
            channels.append((source,quality,qmask,['log_score','severe','reference_agreement68','reference_agreement95']))
    channels.extend([('gaia','qualified',evidence['parallax_eligible'],['log_score','tail']),
        ('class','qualified',evidence['class_label']>=0,['log_score']),
        ('mws_class','qualified',evidence['mws_class_label']>=0,['log_score'])])
    for source,_,qmask,names in channels:
        if qmask.any() and any(source+'/'+metric not in main or source+'/'+metric not in base for metric in names):
            raise ValueError('Qualified evidence requires both complete score channels: '+source)
    rows=[]; summaries=[]; empty=np.full(n,np.nan)
    for role in ['new_sky_field','new_source_seen_sky_field']:
        for cell in range(N_CELLS):
            cell_mask=(photo['role']==role) & (domain['cell']==cell)
            for group,group_mask in groups.items():
                use=cell_mask & group_mask; checks=[]
                for source,quality,qmask,names in channels:
                    mask=use & qmask
                    for metric in names:
                        key=source+'/'+metric
                        check=assess_pair(main.get(key,empty)[mask],base.get(key,empty)[mask],photo['cell'][mask],main_ok[mask],base_ok[mask],metric)
                        row=dict(role=role,cell=cell,group=group,main=MAIN_VIEW,baseline=BASELINE,source=source,
                            qualification=quality,metric=metric,group_n=int(use.sum()),qualified_n=int(mask.sum()),**check)
                        rows.append(row); checks.append(row)
                summaries.append(dict(role=role,cell=cell,group=group,n=int(use.sum()),
                    original_sky_cells=len(np.unique(photo['cell'][use])),
                    main_numerical_failed=int((use & ~main_ok).sum()),**summarize_cell(checks)))
    return rows,summaries


def main():
    require_compute(); root=ROOT/'data/v3_1'; out=root/'domain_assessment'; out.mkdir(exist_ok=True)
    freeze=root/'final_validation/freeze.json'; verify_freeze(ROOT,freeze)
    paths=[freeze,root/'domain_grid/complete.json',root/'final_validation/report/complete.json',
        ROOT/'scripts/v3_1_domain_assessment.py',ROOT/'src/photdis/v3_1/domain_assessment.py',
        ROOT/'docs/v3_1_domain_assessment_protocol.md']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}
    grid_done=json.loads(paths[1].read_text()); path=root/'domain_grid/field.npz'
    assert grid_done['all_PSF'] and not grid_done['references_read'] and sha(path)==grid_done['outputs'][path.name]
    photo,evidence,_,views,metrics=read_cohort(root,'field',inputs); assert MAIN_VIEW in views and BASELINE in views
    marker=out/'complete.json'
    if marker.exists():
        old=json.loads(marker.read_text()); assert old['inputs']==inputs
        for name,digest in old['outputs'].items():assert sha(out/name)==digest
        return
    rows,summaries=field_cell_screens(photo,evidence,metrics,load(path)); outputs=[]
    for role in ['new_sky_field','new_source_seen_sky_field']:
        for source in ['astronn','specdis','spdist','gaia','class','mws_class']:
            name=f'{role}_{source}.csv'; csv_rows(out/name,[r for r in rows if r['role']==role and r['source']==source]); outputs.append(name)
        name=f'{role}_cells.csv'; csv_rows(out/name,[r for r in summaries if r['role']==role]); outputs.append(name)
    save_json(marker,dict(inputs=inputs,outputs={name:sha(out/name) for name in outputs},all_PSF=True,cells=N_CELLS,
        screens=len(rows),cell_summaries=len(summaries),main=MAIN_VIEW,baseline=BASELINE,job=os.getenv('SLURM_JOB_ID'),
        scientific_domain_adjudication_complete=False,
        scope='Same-ID finalcandidate-minus-zero relative reference screens. Mean guards, sky intervals, sparse/missing evidence and numerical flags remain separate. No absolute calibration certificate; final domain adjudication additionally requires radial, common-scale and finite-response support evidence.'))


if __name__=='__main__':main()

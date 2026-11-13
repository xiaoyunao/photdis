#!/usr/bin/env python3
"""Freeze matched target/seen-development probes before response scoring."""
import hashlib
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute


def main():
    require_compute();old=ROOT/'data/v3_0';root=ROOT/'data/v3_1';base=root/'diagnostics/diagnostic';out=root/'response_probes';out.mkdir(exist_ok=True)
    paths=[old/'cohorts/application.npz',old/'cohorts/development.npz',old/'evaluation/development.npz',base/'complete.json',
        root/'target_native_pilot/complete.json',root/'target_native_pilot/roles.npz',old/'calibration.npz',
        ROOT/'scripts/v3_1_response_probe_prepare.py',ROOT/'docs/v3_1_response_probe_protocol.md']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    marker=out/'complete.json'
    if marker.exists():
        d=json.loads(marker.read_text());assert d['inputs']==inputs
        for n,h in d['outputs'].items():assert sha(out/n)==h
        return
    target=load(paths[0]);development=load(paths[1]);evidence=load(paths[2]);roles=load(paths[5])
    np.testing.assert_array_equal(target['identity'],roles['identity']);np.testing.assert_array_equal(development['identity'],evidence['identity'])
    done=json.loads(paths[3].read_text());assert sha(paths[1])==done['inputs'][str(paths[1].relative_to(ROOT))]
    assert sha(paths[2])==done['inputs'][str(paths[2].relative_to(ROOT))]
    metadata=[]
    for name,digest in done['objects'].items():
        p=base/name;assert sha(p)==digest
        rr=[r for r in json.loads(p.read_text())['results'] if r['control']=='response12'];assert len(rr)==1;metadata.append(rr[0])
    selected=[];design=[]
    for c in (-1,0,1,2):
        group=[r for r in metadata if r['class_label']==c]
        ranked=sorted(group,key=lambda r:hashlib.sha256(f"320001:{r['identity']}".encode()).digest())[:64]
        for r in ranked:
            selected.append(r['row']);design.append(dict(identity=r['identity'],row=r['row'],reference_class=c,cell=r['cell'],
                original_inclusion_probability=r['inclusion_probability'],conditional_probe_probability=len(ranked)/len(group),
                inclusion_probability=r['inclusion_probability']*len(ranked)/len(group),source_marker=f"object_{r['identity']}.json"))
    selected=np.asarray(selected,int);rank=sorted(range(len(target['identity'])),key=lambda i:hashlib.sha256(f"319802:{int(target['identity'][i])}".encode()).digest())[:128]
    assert set(rank)<=set(roles['pilot_rows'])
    for label,table,rows in [('target',target,np.array(rank)),('development',development,selected)]:
        assert (table['TYPE'][rows]=='PSF').all() and len(np.unique(table['identity'][rows]))==len(rows)
        arrays={k:v[rows] for k,v in table.items()};arrays['original_rows']=rows
        if label=='target':arrays['candidate']=roles['candidate'][rows]
        atomic_npz(out/f'{label}_photometry.npz',**arrays)
    assert (evidence['TYPE'][selected]=='PSF').all();atomic_npz(out/'development_evidence.npz',**{k:v[selected] for k,v in evidence.items()})
    save_json(out/'development_design.json',design)
    save_json(marker,dict(inputs=inputs,outputs={n:sha(out/n) for n in ['target_photometry.npz','development_photometry.npz','development_evidence.npz','development_design.json']},
        target_rows=128,development_rows=len(selected),all_PSF=True,missing_TYPE_excluded=0,non_PSF_excluded=0,
        job=os.getenv('SLURM_JOB_ID'),response_scores_opened=False,new_validation_outcomes_opened=False,
        scope='Fixed128target and stratified seen-development response probes. Existing development labels only; no new validation. Probe predictive evidence differs from population-marginal evidence.'))


if __name__=='__main__':main()

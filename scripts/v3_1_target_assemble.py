#!/usr/bin/env python3
"""Authenticate and assemble target matrices without dropping failed contexts."""
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute


def main():
    require_compute();root=ROOT/'data/v3_1';native=root/'target_native';entry=root/'target_entry';out=root/'target_matrices';out.mkdir(exist_ok=True)
    roles_path=root/'target_native_pilot/roles.npz';roles=load(roles_path);n=len(roles['identity']);assert n==26000 and (roles['TYPE']=='PSF').all()
    paths=[roles_path,root/'target_native_pilot/complete.json',root/'luminosity_fit_refined/parameters.npz',
        ROOT/'scripts/v3_1_target_assemble.py',ROOT/'docs/v3_1_target_transfer_protocol.md',ROOT/'docs/v3_1_target_entry_protocol.md']
    paths += [native/f'shard_{i}/complete.json' for i in range(4)]+[entry/f'shard_{i:02}/complete.json' for i in range(16)]
    missing=[str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:save_json(out/'progress.json',dict(missing=missing));raise RuntimeError('Matrix shards incomplete; no final assembly')
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    native_done=[json.loads((native/f'shard_{i}/complete.json').read_text()) for i in range(4)]
    entry_done=[json.loads((entry/f'shard_{i:02}/complete.json').read_text()) for i in range(16)]
    assert all(d['all_PSF'] for d in native_done+entry_done)
    assert all(d['inputs']==native_done[0]['inputs'] for d in native_done)
    assert all(d['inputs']==entry_done[0]['inputs'] for d in entry_done)
    for d in native_done+entry_done:assert d['inputs'][str(roles_path.relative_to(ROOT))]==sha(roles_path)
    shape=(n,34,3,5);response=np.empty(shape);broad=np.empty(shape);candidate=np.full(shape,np.nan)
    fine=np.full(shape,np.nan);scale=np.empty(n);audited=np.zeros(n,bool);flags=[];source_markers={}
    errors_broad=np.empty(shape);errors_candidate=np.full(shape,np.nan)
    second_broad=np.empty(shape);second_candidate=np.full(shape,np.nan)
    previous_broad=np.empty(shape);previous_candidate=np.full(shape,np.nan)
    for row,identity in enumerate(roles['identity']):
        name=f'object_{int(identity)}.json';path=native/name;nd=native_done[row%4]
        assert sha(path)==nd['objects'][name];record=json.loads(path.read_text());assert record['inputs']==nd['inputs']
        assert sha(path.with_suffix('.npz'))==record['sha256'];data=load(path.with_suffix('.npz'))
        assert data['identity'][0]==identity and data['TYPE'][0]=='PSF' and data['row'][0]==row
        response[row]=data['coarse_response'];scale[row]=data['logscale'];audited[row]=record['fine_audited']
        if audited[row]:fine[row]=data['fine_response']
        flags.append(dict(row=row,identity=int(identity),kind='native',passed=record['passed']))
        source_markers[str(path.relative_to(ROOT))]=sha(path)
        for label,dest,errors in [('broad',broad,errors_broad)]+([('candidate',candidate,errors_candidate)] if roles['candidate'][row] else []):
            folder=entry/label/f'object_{int(identity)}';marker=folder/'complete.json';r=json.loads(marker.read_text())
            assert r['inputs']==entry_done[row%16]['inputs'] and r['identity']==identity and r['row']==row and r['TYPE']=='PSF'
            for filename,digest in r['outputs'].items():assert sha(folder/filename)==digest
            draws=r['selected_draws'];value=load(folder/f'q{draws}_seed319711.npz');difference=load(folder/f'q{draws}_differences.npz')
            dest[row]=value['alpha'];errors[row]=np.maximum(difference['independent'],difference['previous'])+value['pruning_bound']
            second=second_candidate if label=='candidate' else second_broad
            previous=previous_candidate if label=='candidate' else previous_broad
            assert len(r['attempts'])>=2
            second[row]=load(folder/f'q{draws}_seed319712.npz')['alpha']
            previous_draws=r['attempts'][-2]['draws']
            previous[row]=load(folder/f'q{previous_draws}_seed319711.npz')['alpha']
            flags.append(dict(row=row,identity=int(identity),kind=label,passed=r['passed']))
            source_markers[str(marker.relative_to(ROOT))]=sha(marker)
        if row%1024==0:print('ASSEMBLE TARGET',row,flush=True)
    assert np.isfinite(response).all() and np.isfinite(broad).all() and np.isfinite(scale).all()
    assert np.isfinite(candidate[roles['candidate']]).all()
    atomic_npz(out/'matrices.npz',**roles,response=response,fine_response=fine,fine_audited=audited,logscale=scale,
        selection_broad=broad,selection_candidate=candidate,selection_error_broad=errors_broad,selection_error_candidate=errors_candidate,
        selection_second_broad=second_broad,selection_second_candidate=second_candidate,
        selection_previous_broad=previous_broad,selection_previous_candidate=previous_candidate,
        correction_ids=np.arange(34))
    csv_rows(out/'status.csv',flags);save_json(out/'source_markers.json',source_markers)
    save_json(out/'complete.json',dict(inputs=inputs,outputs={name:sha(out/name) for name in ['matrices.npz','status.csv','source_markers.json']},
        all_PSF=True,rows=n,candidate_count=int(roles['candidate'].sum()),fine_audited=int(audited.sum()),
        passed=all(r['passed'] for r in flags),failed=sum(not r['passed'] for r in flags),all_rows_retained=True,
        job=os.getenv('SLURM_JOB_ID'),scope='Authenticated native/selection matrix assembly. All numerical failures retained; failed assembly blocks fitting, never drops sample rows.'))


if __name__=='__main__':main()

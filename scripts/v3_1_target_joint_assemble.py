#!/usr/bin/env python3
"""Repack authenticated joint sidecars by correction to avoid repeated large file scans."""
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute


def main():
    require_compute();root=ROOT/'data/v3_1';source=root/'target_joint';out=root/'target_joint_matrices';out.mkdir(exist_ok=True)
    paths=[root/'target_native_pilot/roles.npz',root/'target_native_pilot/complete.json',ROOT/'scripts/v3_1_target_joint_assemble.py']
    paths += [source/f'shard_{i}/complete.json' for i in range(4)]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};roles=load(paths[0]);n=len(roles['identity'])
    assert sha(paths[0])==json.loads(paths[1].read_text())['outputs']['roles.npz'] and n==26000 and (roles['TYPE']=='PSF').all()
    done=[json.loads(p.read_text()) for p in paths[3:]];assert all(d['passed'] and d['all_PSF'] for d in done)
    freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    marker=out/'complete.json'
    if marker.exists():
        old=json.loads(marker.read_text());assert old['inputs']==inputs
        for name,digest in old['outputs'].items():assert sha(out/name)==digest
        return
    coarse=np.empty((34,n,3,5,6));fine=[];fine_rows=[];scale=np.empty(n);sources={}
    for row,identity in enumerate(roles['identity']):
        name=f'object_{int(identity)}.json';p=source/name;assert sha(p)==done[row%4]['objects'][name]
        record=json.loads(p.read_text());assert record['inputs']==done[row%4]['inputs'] and record['passed']
        payload=p.with_suffix('.npz');assert sha(payload)==record['sha256'];value=load(payload)
        assert value['identity'][0]==identity and value['row'][0]==row and value['TYPE'][0]=='PSF'
        coarse[:,row]=value['coarse_joint_response'];scale[row]=value['logscale']
        if record['fine_audited']:fine_rows.append(row);fine.append(value['fine_joint_response'])
        sources[str(p.relative_to(ROOT))]=sha(p)
        if row%2048==0:print('JOINT REPACK',row,flush=True)
    assert len(fine_rows)==258
    fine=np.stack(fine,axis=1);fine_rows=np.array(fine_rows);assert np.isfinite(coarse).all() and np.isfinite(fine).all() and (coarse>=0).all() and (fine>=0).all()
    atomic_npz(out/'metadata.npz',identity=roles['identity'],TYPE=roles['TYPE'],logscale=scale,fine_rows=fine_rows,correction_ids=np.arange(34))
    for correction in range(34):
        atomic_npz(out/f'correction_{correction:02}.npz',correction=np.array(correction),coarse_joint_response=coarse[correction],fine_joint_response=fine[correction])
    save_json(out/'source_markers.json',sources)
    save_json(marker,dict(inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.suffix=='.npz' or p.name=='source_markers.json'},
        n=n,corrections=34,fine_audited=258,all_PSF=True,passed=True,job=os.getenv('SLURM_JOB_ID'),
        scope='Exact float64 array repack only; original26000joint sidecars preserved. No scientific fitting or probability renormalization.'))


if __name__=='__main__':main()

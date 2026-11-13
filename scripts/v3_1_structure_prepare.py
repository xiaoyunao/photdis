#!/usr/bin/env python3
"""Freeze a seen-system, photo-only structure demonstration sample."""
import hashlib
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute


def main():
    require_compute();old=ROOT/'data/v3_0/independent_scale';out=ROOT/'data/v3_1/structure';out.mkdir(exist_ok=True)
    paths=[old/'photometry.npz',old/'photometry.json',ROOT/'configs/v3_0_independent_scale.json',
        ROOT/'docs/v3_1_structure_protocol.md',ROOT/'scripts/v3_1_structure_prepare.py']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    marker=out/'complete.json'
    if marker.exists():
        d=json.loads(marker.read_text());assert d['inputs']==inputs and sha(out/'photometry.npz')==d['outputs']['photometry.npz'];return
    admission=json.loads(paths[1].read_text());assert sha(paths[0])==admission['sha256'] and sha(paths[2])==admission['config_sha256']
    table=load(paths[0]);assert len(table['identity'])==8192 and (table['TYPE']=='PSF').all()
    rows=np.array(sorted(range(len(table['identity'])),key=lambda i:hashlib.sha256(f"320201:{int(table['identity'][i])}".encode()).digest())[:512])
    arrays={k:v[rows] for k,v in table.items()};arrays['original_rows']=rows
    arrays['conditional_probe_probability']=np.full(512,512/8192)
    arrays['design_weight']=table['design_weight'][rows]*8192/512
    assert (arrays['TYPE']=='PSF').all() and len(np.unique(arrays['identity']))==512 and (arrays['radius_deg']<=1).all()
    atomic_npz(out/'photometry.npz',**arrays)
    save_json(marker,dict(inputs=inputs,outputs={'photometry.npz':sha(out/'photometry.npz')},n=512,all_PSF=True,
        original_admission=admission,subset_non_PSF_excluded=0,subset_missing_TYPE_excluded=0,job=os.getenv('SLURM_JOB_ID'),
        membership_used_for_selection=False,new_validation_outcomes_opened=False,
        scope='SeenNGC288 photographic-first uniform identity subsample. Existing shared dynamical scale will be a prior, not validation truth; counts concern512observed objects only.'))


if __name__=='__main__':main()

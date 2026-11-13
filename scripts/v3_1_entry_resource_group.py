#!/usr/bin/env python3
"""Run disjoint existing selection shards concurrently within one allocated node."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from v2_1_near import ROOT,sha,save_json
from photdis.stage1 import require_compute


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--shards',nargs='+',type=int,required=True);a=p.parse_args()
    assert 1<=len(a.shards)<=4 and len(set(a.shards))==len(a.shards) and all(0<=s<16 for s in a.shards)
    job=os.environ['SLURM_JOB_ID'];out=ROOT/f'data/v3_1/target_entry_resource_groups/job_{job}';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'scripts/v3_1_target_entry.py';locks=ROOT/'data/v3_1/target_entry_resource_groups/locks';locks.mkdir(exist_ok=True)
    handles=[];streams=[];processes=[];records=[]
    inputs=dict(science_script_sha256=sha(source),resource_script_sha256=sha(ROOT/'scripts/v3_1_entry_resource_group.py'),shards=a.shards)
    freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    try:
        for shard in a.shards:
            handle=(locks/f'shard_{shard:02}.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(handle)
        for shard in a.shards:
            log=ROOT/f'logs/photdis_v3_1_entrygroup_{job}_shard{shard:02}.out';stream=log.open('x');streams.append(stream)
            child=subprocess.Popen([sys.executable,str(source),'--shard',str(shard)],stdout=stream,stderr=subprocess.STDOUT)
            processes.append((shard,child,log))
        for shard,child,log in processes:
            code=child.wait();record=dict(shard=shard,exit_code=code,log=str(log.relative_to(ROOT)))
            if code==0:
                marker=ROOT/f'data/v3_1/target_entry/shard_{shard:02}/complete.json';result=json.loads(marker.read_text())
                assert result['shard']==shard and result['all_PSF'] and result['inputs']['scripts/v3_1_target_entry.py']==inputs['science_script_sha256']
                record.update(marker=str(marker.relative_to(ROOT)),marker_sha256=sha(marker),numerically_passed=result['passed'])
            records.append(record)
    finally:
        for stream in streams:stream.close()
        for handle in handles:handle.close()
    save_json(out/'complete.json',dict(inputs=inputs,results=records,job=job,all_exited_successfully=all(r['exit_code']==0 for r in records),
        scope='Resource grouping only, up to4disjoint shards x32existing workers. Original program/inputs/seeds/objects/sidecars unchanged; no duplicate pending/running writer may be scheduled.'))
    if any(r['exit_code'] for r in records):raise RuntimeError('Resource group has a failed child; preserve logs and checkpoints')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Run only preflighted pending response stages sequentially with original code."""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from v2_1_near import ROOT,sha,save_json
from photdis.stage1 import require_compute


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--plan',required=True)
    p.add_argument('--group',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',-1)))
    a=p.parse_args();path=ROOT/a.plan;plan=json.loads(path.read_text())
    assert 0<=a.group<len(plan['groups']) and len(os.sched_getaffinity(0))>=32
    for name,digest in plan['sources'].items():assert sha(ROOT/name)==digest
    for row in plan['stages']:
        if row['complete']:assert sha(ROOT/row['path'])==row['sha256']
    group=plan['groups'][a.group];assert 1<=len(group)<=8
    out=ROOT/'data/v3_1/response_probes/resource_execution'/f'group_{a.group:02}'
    out.mkdir(parents=True,exist_ok=True);marker=out/'complete.json'
    if marker.exists():raise RuntimeError('Resource group already complete; preserve its provenance')
    record=dict(plan_sha256=sha(path),group=a.group,indices=group,job=os.getenv('SLURM_JOB_ID'),
        original_stage_records=[],execution_complete=False,
        scope='Sequential unchanged original response drivers, at most32workers. No new scientific input, seed, grid, numerical gate, normalization or completed-product rewrite.')
    start=time.monotonic()
    for index in group:
        stages=[r for r in plan['stages'] if r['index']==index]
        assert [r['stage'] for r in stages]==['probe','refine']
        for row in stages:
            target=ROOT/row['path']
            if row['complete']:
                assert sha(target)==row['sha256'];continue
            # A completed stage cannot be restarted through this resource plan.
            if target.exists():raise RuntimeError(f'Unexpected completed stage: {target}')
            script=plan['stage_scripts'][row['stage']]
            assert sha(ROOT/script)==plan['sources'][script]
            subprocess.run([sys.executable,'-u',str(ROOT/script),'--index',str(index)],check=True,cwd=ROOT)
            done=json.loads(target.read_text())
            assert done['index']==index and done['global_draw_id']==row['draw'] and done['all_PSF']
            record['original_stage_records'].append(dict(index=index,stage=row['stage'],path=row['path'],sha256=sha(target)))
            save_json(out/'progress.json',record)
            print('EXACT RESPONSE STAGE COMPLETE',index,row['stage'],flush=True)
    for row in plan['stages']:
        if row['complete']:assert sha(ROOT/row['path'])==row['sha256']
    record.update(execution_complete=True,elapsed_seconds=time.monotonic()-start,
        maximum_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    save_json(marker,record)


if __name__=='__main__':main()

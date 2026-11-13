#!/usr/bin/env python3
"""Metadata-only, fail-closed replacement of pending short response jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'scripts/v3_1_response_probe.py':'7b0070af2daf00217a635d2c3488c63a7d545b96eb01ee687c4db1eb6dbe6be3',
    'scripts/v3_1_response_probe_refine.py':'977fbee68315be96f4ad5527933412827bcfce2797d0b6b30a143409a7ebf58b'}
NAMES={'photdis_v3_1_responseprobe':'probe','photdis_v3_1_proberefine':'refine'}


def run(args):
    return subprocess.check_output(args,text=True,cwd=ROOT).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def queue():
    return [dict(zip(['id','name','state','qos'],line.split('|'))) for line in
        run(['squeue','-r','-u','yunao','-h','-o','%i|%j|%T|%q']).splitlines()]


def expanded(job):
    if '_[' not in job:return [job]
    base,tail=job.split('_[',1);tail=tail.rstrip(']').split('%')[0];ids=[]
    for term in tail.split(','):
        a,_,b=term.partition('-');ids.extend(f'{base}_{i}' for i in range(int(a),int(b or a)+1))
    return ids


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--commit',required=True)
    a=p.parse_args();assert run(['git','rev-parse','HEAD'])==a.commit
    assert not run(['git','status','--porcelain','--untracked-files=no'])
    for name,value in EXPECTED.items():assert digest(ROOT/name)==value
    sources={name:digest(ROOT/name) for name in list(EXPECTED)+[
        'scripts/v3_1_response_resource_group.py','scripts/run_v3_1_response_resource_group.sh']}
    stages=[]
    for index in range(64):
        draw=index if index<32 else index-32+1000
        model=ROOT/(f'data/v3_0/uncertainty_refit/draw_{draw:02}/complete.json' if index<32 else
            f'data/v3_1/response_library/refits/draw_{draw}/complete.json')
        assert json.loads(model.read_text())['converged']
        for stage,folder in [('probe',f'draw_{draw:04}'),('refine',f'development_refined/draw_{draw:04}')]:
            path=ROOT/f'data/v3_1/response_probes/{folder}/complete.json'
            row=dict(index=index,draw=draw,stage=stage,path=str(path.relative_to(ROOT)),complete=path.exists())
            if path.exists():
                done=json.loads(path.read_text());assert done['all_PSF'] and done['global_draw_id']==draw
                row['sha256']=digest(path)
            stages.append(row)
    before=queue();writers={}
    for row in before:
        if row['name'] in NAMES:
            assert '_' in row['id'];key=(int(row['id'].split('_')[1]),NAMES[row['name']])
            assert key not in writers;writers[key]=row
    missing=[r for r in stages if not r['complete']]
    assert missing
    for r in missing:assert (r['index'],r['stage']) in writers
    retained={r['index'] for r in missing if writers[r['index'],r['stage']]['state']!='PENDING'}
    selected=[r for r in missing if r['index'] not in retained]
    indices=sorted({r['index'] for r in selected});assert indices
    groups=[indices[i:i+8] for i in range(0,len(indices),8)]
    old=[writers[r['index'],r['stage']]['id'] for r in selected]
    keep=[writers[r['index'],r['stage']]['id'] for r in missing if r['index'] in retained]
    plan=dict(source_commit=a.commit,sources=sources,stage_scripts=dict(probe='scripts/v3_1_response_probe.py',
        refine='scripts/v3_1_response_probe_refine.py'),stages=stages,groups=groups,old_pending=old,
        retained_dependencies=keep,scope='At most8indices per node, sequential unchanged original stages. Completed stages are hash-checked and never rerun. All64states still required by original report.')
    summary=dict(groups=groups,pending_stages=len(old),retained_live_indices=sorted(retained),completed_stages=128-len(missing))
    if not a.execute:print(json.dumps(summary));return
    folder=ROOT/'data/v3_1/response_probes/resource_plan';folder.mkdir(exist_ok=True)
    journal=folder/'submission.json';path=folder/'plan.json'
    assert not journal.exists() and not path.exists()
    def save():journal.write_text(json.dumps(plan,indent=2)+'\n')
    report=next(r for r in before if r['id']=='61221949');assert report['state']=='PENDING'
    # Recheck every scheduled writer immediately before state-restricted cancellation.
    latest={r['id']:r for r in queue()}
    assert all(latest[j]['state']=='PENDING' for j in old)
    save();plan['cancel_command']=['scancel','--state=PENDING',*old];save();run(plan['cancel_command'])
    assert not set(old)&{r['id'] for r in queue()}
    account=run(['sacct','-S','2026-11-12','--array','-X','-n','-P','-j',','.join(sorted({j.split('_')[0] for j in old})),
        '-o','JobID,State,ElapsedRaw,AllocNodes'])
    states={}
    for line in account.splitlines():
        job,state,elapsed,nodes=line.split('|')[:4]
        for key in expanded(job):states[key]=(state,elapsed,nodes)
    plan['cancelled_receipts']={}
    for job in old:
        if job in states:
            state,elapsed,nodes=states[job];assert state.startswith('CANCELLED') and int(elapsed)==int(nodes)==0
            plan['cancelled_receipts'][job]=states[job]
        else:
            receipt=run(['scontrol','show','job',job,'-o'])
            assert re.search(r'\bJobState=CANCELLED\b',receipt) and 'RunTime=00:00:00' in receipt and 'AllocTRES=(null)' in receipt
            plan['cancelled_receipts'][job]=receipt
        save()
    for r in stages:
        target=ROOT/r['path']
        if r['complete']:assert digest(target)==r['sha256']
        elif r['index'] not in retained:assert not target.exists()
    path.write_text(json.dumps({k:v for k,v in plan.items() if k not in ['cancelled_receipts','cancel_command']},indent=2)+'\n')
    command=['sbatch','--parsable',f'--array=0-{len(groups)-1}%2','scripts/run_v3_1_response_resource_group.sh','--plan',str(path.relative_to(ROOT))]
    plan['submit_command']=command;save();job=run(command).split(';')[0];assert job.isdigit()
    plan['replacement_array']=job;save()
    # The original report still authenticates every one of64probe/refinement states.
    # Remove dependencies on intentionally cancelled array parents, retaining live writers.
    assert next(r for r in queue() if r['id']=='61221949')['state']=='PENDING'
    deps='afterok:'+':'.join([job]+keep);run(['scontrol','update','JobId=61221949','Dependency='+deps])
    plan['report_dependency']=deps;plan['report_controller']=run(['scontrol','show','job','61221949','-o']);save()
    print(json.dumps(dict(summary,replacement_array=job,report_dependency=deps)))


if __name__=='__main__':main()

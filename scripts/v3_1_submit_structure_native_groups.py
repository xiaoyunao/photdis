#!/usr/bin/env python3
"""Metadata-only replacement of unstarted native structure stage pairs."""
import argparse
import json
import subprocess
import time
from v3_1_submit_response_groups import ROOT,run,digest,queue,expanded

EXPECTED={'scripts/v3_1_structure_native.py':'5fe6d0e750e57281685fbbf48f728b90498a33caf3c39868bb8cda6296775825',
    'scripts/v3_1_structure_native_refine.py':'8eb86154d7809c86a35e81b443bd60e04b5abf9e4986462a4bb565a4e07c604b'}
NAMES={'photdis_v3_1_structurenative':'native','photdis_v3_1_structurerefine':'native_refined'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--commit',required=True)
    a=p.parse_args();assert run(['git','rev-parse','HEAD'])==a.commit
    assert not run(['git','status','--porcelain','--untracked-files=no'])
    for name,value in EXPECTED.items():assert digest(ROOT/name)==value
    sources={name:digest(ROOT/name) for name in list(EXPECTED)+[
        'scripts/v3_1_structure_native_resource_group.py','scripts/run_v3_1_structure_native_resource_group.sh']}
    stages=[]
    for index in range(32):
        model=ROOT/f'data/v3_0/uncertainty_refit/draw_{index:02}/complete.json'
        assert json.loads(model.read_text())['converged']
        for stage in ['native','native_refined']:
            path=ROOT/f'data/v3_1/structure/{stage}/draw_{index:02}/complete.json'
            row=dict(index=index,draw=index,stage=stage,path=str(path.relative_to(ROOT)),complete=path.exists())
            if path.exists():
                done=json.loads(path.read_text());assert done['all_PSF'] and done['response_draw']==index and done['n']==512
                row['sha256']=digest(path)
            stages.append(row)
    before=queue();writers={}
    for row in before:
        if row['name'] in NAMES:
            key=(int(row['id'].split('_')[1]),NAMES[row['name']]);assert key not in writers;writers[key]=row
    missing=[r for r in stages if not r['complete']];assert missing
    for r in missing:assert (r['index'],r['stage']) in writers
    retained={r['index'] for r in missing if writers[r['index'],r['stage']]['state']!='PENDING'}
    selected=[r for r in missing if r['index'] not in retained]
    indices=sorted({r['index'] for r in selected});assert indices
    groups=[indices[i:i+7] for i in range(0,len(indices),7)]
    old=[writers[r['index'],r['stage']]['id'] for r in selected]
    parents=sorted({j.split('_')[0] for j in old})
    # No downstream job may silently remain attached to a cancelled parent.
    downstream=[]
    for line in run(['squeue','-r','-u','yunao','-h','-o','%i|%E']).splitlines():
        job,dependency=line.split('|',1)
        if any(parent in dependency for parent in parents) and job not in old:downstream.append((job,dependency))
    assert not downstream,downstream
    plan=dict(source_commit=a.commit,sources=sources,stage_scripts=dict(native='scripts/v3_1_structure_native.py',
        native_refined='scripts/v3_1_structure_native_refine.py'),stages=stages,groups=groups,old_pending=old,
        retained_live_indices=sorted(retained),scope='At most7native/refined stage pairs per node, sequential unchanged32-worker original drivers. Completed products and all numerical failures retained. Entry integrals and running tasks untouched; final assembly still requires all32states.')
    summary=dict(groups=groups,pending_stages=len(old),retained_live_indices=sorted(retained),completed_stages=64-len(missing))
    if not a.execute:print(json.dumps(summary));return
    folder=ROOT/'data/v3_1/structure/native_resource_plan';folder.mkdir(exist_ok=True)
    journal=folder/'submission.json';path=folder/'plan.json';assert not journal.exists() and not path.exists()
    def save():journal.write_text(json.dumps(plan,indent=2)+'\n')
    latest={r['id']:r for r in queue()};assert all(latest[j]['state']=='PENDING' for j in old)
    plan['cancel_command']=['scancel','--state=PENDING',*old];save();run(plan['cancel_command'])
    assert not set(old)&{r['id'] for r in queue()}
    # Accounting is asynchronous; retain every observation and retry metadata only.
    # Never repeat cancellation or submit while terminal/no-allocation proof is absent.
    plan['verification_attempts']=[];receipts={}
    for attempt in range(4):
        account=run(['sacct','-S','2026-11-12','--array','-X','-n','-P','-j',','.join(parents),'-o','JobID,State,ElapsedRaw,AllocNodes'])
        states={}
        for line in account.splitlines():
            job,state,elapsed,nodes=line.split('|')[:4]
            for key in expanded(job):states[key]=(state,elapsed,nodes)
        pending={}
        for job in old:
            if job in receipts:continue
            values=states.get(job)
            if values and values[0].startswith('CANCELLED') and int(values[1])==int(values[2])==0:
                receipts[job]=values;continue
            result=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True)
            text=result.stdout
            if result.returncode==0 and 'JobState=CANCELLED' in text and 'RunTime=00:00:00' in text and 'AllocTRES=(null)' in text:
                receipts[job]=text
            else:pending[job]=dict(accounting=values,controller_returncode=result.returncode,controller=text)
        plan['verification_attempts'].append(dict(attempt=attempt,unverified=pending));plan['cancelled_receipts']=receipts;save()
        if not pending:break
        if attempt<3:time.sleep(5)
    assert set(receipts)==set(old),'Missing cancellation proof; inspect retained journal before recovery'
    assert not set(old)&{r['id'] for r in queue()}
    for row in stages:
        target=ROOT/row['path']
        if row['complete']:assert digest(target)==row['sha256']
        elif row['index'] not in retained:assert not target.exists()
    path.write_text(json.dumps({k:v for k,v in plan.items() if k not in ['verification_attempts','cancelled_receipts','cancel_command']},indent=2)+'\n')
    command=['sbatch','--parsable',f'--array=0-{len(groups)-1}%2','scripts/run_v3_1_structure_native_resource_group.sh','--plan',str(path.relative_to(ROOT))]
    plan['submit_command']=command;save();job=run(command).split(';')[0];assert job.isdigit();plan['replacement_array']=job;save()
    print(json.dumps(dict(summary,replacement_array=job)))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Run independent imaging-cache shards and native random scans in one allocation."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
from photdis.stage1 import require_compute
from halo_common import *


def run(item):
    stage,args,tag=item
    log=folder('execution')/(tag+'_'+os.getenv('SLURM_JOB_ID','local')+'.log')
    with log.open('wb') as f:
        r=subprocess.run([sys.executable,str(Path(__file__).with_name('halo_'+stage+'.py')),*args],
                         stdout=f,stderr=subprocess.STDOUT)
    print(tag,r.returncode,flush=True)
    return dict(stage=tag,exit_code=r.returncode,log=str(log.relative_to(ROOT)))


if __name__=='__main__':
    require_compute()
    jobs=[('randoms',[side],'area_'+side) for side in ('north','south')]
    jobs += [('cache',['scan','--shard',str(i)],f'cache_{i:02}') for i in range(C['cache_shards'])]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(run,jobs))
    save_json(folder('execution')/'cache_batch.json',dict(job=os.getenv('SLURM_JOB_ID'),results=results))
    if any(r['exit_code'] for r in results):raise SystemExit(1)

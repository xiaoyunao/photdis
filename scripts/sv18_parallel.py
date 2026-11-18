#!/usr/bin/env python3
"""Run independent small science stages within one allocated CPU node."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

from photdis.stage1 import require_compute
from sv18_common import DEST, out

def run(stage):
    folder=out('execution')
    log=folder/(stage+'.log')
    script=Path(__file__).with_name('sv18_'+stage+'.py')
    with log.open('wb') as handle:
        result=subprocess.run([sys.executable,str(script)],stdout=handle,stderr=subprocess.STDOUT)
    print(stage,result.returncode,flush=True)
    return result.returncode

def main():
    require_compute()
    stages=sys.argv[1:]
    allowed={'populations','benchmark','provenance','systems','predictive','noise'}
    if not stages or not set(stages)<=allowed:
        raise ValueError('Expected independent registered stage names')
    workers=min(3,max(1,int(os.getenv('SLURM_CPUS_PER_TASK','1'))//8))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results=list(executor.map(run,stages))
    if any(results):
        raise SystemExit(1)

if __name__=='__main__':
    main()

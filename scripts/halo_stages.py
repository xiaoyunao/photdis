#!/usr/bin/env python3
"""Dispatch independent fixed scientific controls on allocated CPU cores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from halo_driver import run
from halo_common import *
from photdis.stage1 import require_compute

if __name__=='__main__':
    require_compute()
    p=argparse.ArgumentParser();p.add_argument('stages',nargs='+');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    with ThreadPoolExecutor(args.workers) as pool:
        r=list(pool.map(run,[(s,[],s) for s in args.stages]))
    save_json(folder('execution')/('batch_'+('_'.join(args.stages))+'.json'),dict(results=r,job=os.getenv('SLURM_JOB_ID')))
    if any(q['exit_code'] for q in r):raise SystemExit(1)

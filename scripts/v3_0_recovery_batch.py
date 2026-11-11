#!/usr/bin/env python3
"""Independent catalog recovery processes sharing one allocated CPU node."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
from v2_1_near import ROOT
from photdis.stage1 import require_compute


def main():
    require_compute(); p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=int(os.getenv('SLURM_ARRAY_TASK_ID', 0)))
    p.add_argument('--view', choices=['grz', 'multiband'], default='multiband')
    p.add_argument('--draws', type=int, default=512); p.add_argument('--warmup', type=int, default=512)
    p.add_argument('--workers', type=int, default=8); a = p.parse_args()
    if not 0 <= a.batch < 14: raise ValueError('Fourteen frozen scenarios only')
    def run(catalog):
        command = [sys.executable, '-u', str(ROOT/'scripts/v3_0_recovery.py'), '--catalog', str(catalog),
                   '--view', a.view, '--draws', str(a.draws), '--warmup', str(a.warmup), '--engine', 'cached']
        r = subprocess.run(command)
        if r.returncode: raise RuntimeError('Catalog recovery failed: '+str(catalog))
        return catalog
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        complete = list(pool.map(run, range(a.batch*8, (a.batch+1)*8)))
    print('RECOVERY BATCH', a.view, a.batch, complete, flush=True)


if __name__ == '__main__': main()

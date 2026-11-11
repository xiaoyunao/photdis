#!/usr/bin/env python3
"""Run eight catalog fits for one shared response in one Slurm allocation."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
from v2_1_near import ROOT
from photdis.stage1 import require_compute


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',0)))
    p.add_argument('--profiles',type=int);p.add_argument('--quadrature',type=int)
    p.add_argument('--draws',type=int,default=2048);a=p.parse_args();draw,batch=divmod(a.index,14)
    if not 0<=draw<32:raise ValueError('Only frozen 32 shared draws')
    def run(catalog):
        extra=[]
        profiles=a.profiles or (1024 if draw in (6,14,23,28) else None)
        quadrature=a.quadrature or (512 if draw in (6,14,23,28) else None)
        if profiles:extra += ['--profiles',str(profiles)]
        if quadrature:extra += ['--quadrature',str(quadrature)]
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/v3_0_shared_recovery.py'),'--draw',str(draw),'--catalog',str(catalog),'--draws',str(a.draws),'--production',*extra],check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(run,range(batch*8,(batch+1)*8)))
    print('SHARED RECOVERY BATCH',draw,batch,flush=True)


if __name__=='__main__':main()

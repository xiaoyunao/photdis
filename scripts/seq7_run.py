#!/usr/bin/env python3
"""Run registered controls on an allocated node with retained attempt logs."""
import argparse,subprocess
from datetime import datetime,timezone
from seq7_common import *
from photdis.stage1 import require_compute


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--from-stage',choices=('prepare','predict','curves','evaluate','examples','figures','validate','publish'),default='prepare');args=p.parse_args();stages=['prepare','predict','curves','evaluate','examples','figures','validate','publish'];stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');d=folder('execution')
    for stage in stages[stages.index(args.from_stage):]:
        logfile=d/f'{stamp}_{stage}.log';print('RUN',stage,str(logfile),flush=True)
        with logfile.open('w') as f:subprocess.run([sys.executable,str(ROOT/'scripts'/f'seq7_{stage}.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
    save_json(d/f'{stamp}_run.json',dict(job=os.getenv('SLURM_JOB_ID'),config=CH,from_stage=args.from_stage,completed=True));print('V7_FIXED_CONTROLS_DELIVERED',flush=True)


if __name__=='__main__':main()

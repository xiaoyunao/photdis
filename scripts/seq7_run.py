#!/usr/bin/env python3
"""Run registered controls on an allocated node with retained attempt logs."""
import argparse,subprocess
import shutil
from datetime import datetime,timezone
from seq7_common import *
from photdis.stage1 import require_compute


def main():
    require_compute();stages=['prepare','predict','curves','evaluate','examples','figures','validate','publish'];p=argparse.ArgumentParser();p.add_argument('--from-stage',choices=stages,default='prepare');p.add_argument('--stages',nargs='+',choices=stages);p.add_argument('--archive-existing',action='store_true');args=p.parse_args();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');d=folder('execution');selected=args.stages or stages[stages.index(args.from_stage):]
    for stage in selected:
        if args.archive_existing and (OUT/stage).exists():shutil.copytree(OUT/stage,d/f'{stamp}_{stage}_before')
        logfile=d/f'{stamp}_{stage}.log';print('RUN',stage,str(logfile),flush=True)
        with logfile.open('w') as f:subprocess.run([sys.executable,str(ROOT/'scripts'/f'seq7_{stage}.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
    save_json(d/f'{stamp}_run.json',dict(job=os.getenv('SLURM_JOB_ID'),config=CH,stages=selected,completed=True));print('V7_FIXED_CONTROLS_DELIVERED',flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Run independent field predictions concurrently within one CPU allocation."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import os,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from photdis.stage1 import require_compute


def run(field):
    log=Path('logs')/f"baseline_bulk_{os.environ['SLURM_JOB_ID']}_{field:02}.out"
    with log.open('w') as stream:
        p=subprocess.run([sys.executable,'scripts/baseline_v0_apply.py',str(field)],stdout=stream,stderr=subprocess.STDOUT)
    return field,p.returncode,str(log)


def main():
    require_compute();failed=[];fields=[int(s) for s in sys.argv[1:]] or list(range(14))
    with ThreadPoolExecutor(max_workers=14) as pool:
        for task in as_completed([pool.submit(run,i) for i in fields]):
            field,code,log=task.result();print(field,code,log,flush=True)
            if code:failed.append(field)
    if failed:raise RuntimeError(f'Application fields failed: {failed}')

if __name__=='__main__':main()

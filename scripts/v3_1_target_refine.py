#!/usr/bin/env python3
"""Registered bounded population/count numerical schedule; no scientific retuning."""
import argparse
import json
import os
import subprocess
import sys
from v2_1_near import ROOT,sha,save_json
from photdis.stage1 import require_compute


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--case',choices=['broad','candidate','holdout'],required=True);p.add_argument('--correction',type=int,required=True);a=p.parse_args();assert 0<=a.correction<34
    root=ROOT/'data/v3_1';attempts=[]
    for draws in (512,2048,8192):
        args=['--case',a.case,'--correction',str(a.correction),'--draws',str(draws)]
        subprocess.run([sys.executable,str(ROOT/'scripts/v3_1_target_fit.py'),*args],check=True)
        folder=root/'target_fits'/a.case/f'correction_{a.correction:02}_s{draws}';marker=folder/'complete.json';fit=json.loads(marker.read_text())
        attempt=dict(draws=draws,fit=str(marker.relative_to(ROOT)),fit_sha256=sha(marker),sampler_passed=fit['sampler_passed'],bridge_passed=fit['bridge_passed'],passed=False)
        if fit['sampler_passed']:
            counts=folder/'joint_counts/complete.json'
            if not counts.exists():subprocess.run([sys.executable,str(ROOT/'scripts/v3_1_target_counts.py'),*args],check=True)
            checked=json.loads(counts.read_text());assert checked['inputs'][str(marker.relative_to(ROOT))]==sha(marker)
            assert checked['inputs']['scripts/v3_1_target_counts.py']==sha(ROOT/'scripts/v3_1_target_counts.py')
            for name,digest in checked['outputs'].items():assert sha(counts.parent/name)==digest
            attempt.update(counts=str(counts.relative_to(ROOT)),counts_sha256=sha(counts),counts_passed=checked['passed'],passed=bool(checked['passed'] and fit['bridge_passed']))
        attempts.append(attempt);print('TARGET NUMERICAL SCHEDULE',a.case,a.correction,draws,attempt['passed'],flush=True)
        if attempt['passed']:break
    out=root/'target_refinement'/a.case/f'correction_{a.correction:02}';out.mkdir(parents=True,exist_ok=True)
    save_json(out/'complete.json',dict(case=a.case,correction=a.correction,attempts=attempts,selected_draws=attempts[-1]['draws'],passed=attempts[-1]['passed'],
        script_sha256=sha(ROOT/'scripts/v3_1_target_refine.py'),job=os.getenv('SLURM_JOB_ID'),
        scope='Registered512/2048/8192retained schedule,4chains/fixed512warmup/seeds/priors/selection. All originals and terminal numerical failures retained; no science outcome enters selection.'))


if __name__=='__main__':main()

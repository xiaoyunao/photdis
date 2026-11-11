#!/usr/bin/env python3
"""Run only the verified globally weighted response transfer plan for one field."""
import argparse
import json
import os
import subprocess
import sys
from v2_1_near import ROOT,sha
from photdis.stage1 import require_compute
from v3_0_validation_inputs import check_freeze
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('scope',choices=['catalog','validation'])
    p.add_argument('name',nargs='?',default=os.getenv('SLURM_ARRAY_TASK_ID'));p.add_argument('--workers',type=int,default=64)
    a=p.parse_args();assert a.name is not None
    plan=json.loads((D/'catalog/transfer_plan.json').read_text());assert plan['method_freeze']==check_freeze()
    assert plan['ensemble_sha256']==sha(D/'ensemble_products'/plan['ensemble']/'complete.json')
    for r in plan['responses']:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/v3_0_projection.py'),a.scope,str(a.name),
            '--draw',str(r['draw']),'--draws',str(r['draws']),'--workers',str(a.workers)],check=True)


if __name__=='__main__':main()

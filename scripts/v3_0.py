#!/usr/bin/env python3
"""Unified v3 workflow entry; heavy stages require an allocated compute node.

Examples: v3_0.py status; v3_0.py sed fit; v3_0.py predict development --view grz.
The stage drivers share photdis.v3 likelihood, roles, selection and count code.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    'prepare': 'v3_0_prepare.py', 'calibrate': 'v3_0_calibrate.py',
    'native': 'v3_0_native.py', 'sed': 'v3_0_sed.py',
    'controls': 'v3_0_controls.py', 'cohorts': 'v3_0_cohorts.py',
    'uncertainty': 'v3_0_uncertainty.py', 'benchmark': 'v3_0_response.py',
    'predict': 'v3_0_predict.py', 'independent': 'v3_0_independent.py',
    'selection': 'v3_0_selection.py', 'target': 'v3_0_target.py',
    'qrf': 'v3_0_qrf_predict.py', 'evaluate': 'v3_0_evaluate.py',
    'selection-audit': 'v3_0_selection_audit.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['status']+list(STAGES))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.stage == 'status':
        names = ['preparation.json', 'calibration.json', 'sed_cloud.json',
                 'sed_fit.json', 'controls/complete.json', 'cohorts/complete.json',
                 'uncertainty/complete.json']
        for name in names:
            path = ROOT/'data/v3_0'/name
            if not path.exists():
                path = ROOT/'manifests/v3_0'/name
            if path.exists():
                record = json.loads(path.read_text())
                print(name, json.dumps({k: record[k] for k in
                    ('job', 'jobs', 'n', 'sources', 'admitted', 'converged', 'rows', 'elapsed_seconds') if k in record}))
            else: print(name, 'NO COMPLETION ARTIFACT')
        print('Scientific acceptance and A–F completion require the final delivery audit; file presence is insufficient.')
        return
    from photdis.stage1 import require_compute
    require_compute()
    completed = subprocess.run([sys.executable, '-u', str(ROOT/'scripts'/STAGES[args.stage]), *args.arguments])
    raise SystemExit(completed.returncode)


if __name__ == '__main__': main()

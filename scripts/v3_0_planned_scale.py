#!/usr/bin/env python3
"""Route the independent-scale example through the immutable transfer plan."""
import json
import subprocess
import sys
from v2_1_near import ROOT, sha
from photdis.stage1 import require_compute
from v3_0_validation_inputs import check_freeze


def main():
    require_compute()
    base = ROOT/'data/v3_0'
    plan = json.loads((base/'catalog/transfer_plan.json').read_text())
    assert plan['method_freeze'] == check_freeze()
    assert plan['ensemble_sha256'] == sha(base/'ensemble_products'/plan['ensemble']/'complete.json')
    assert len(plan['responses']) == 1, 'Combine retained responses before scale evaluation'
    response = plan['responses'][0]
    path = base/'validation/independent_scale/population_transfer'/f'refit_{response["draw"]:02}_s{response["draws"]}'
    subprocess.run([sys.executable, str(ROOT/'scripts/v3_0_scale_example.py'), '--target', str(path)], check=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Numerical-only HMC refinement through the unchanged recovery implementation."""
import argparse
import concurrent.futures
import json
import os
import sys
import subprocess
from pathlib import Path
from v2_1_near import ROOT,sha,save_json
from photdis.stage1 import require_compute
import v3_0_shared_recovery as recovery
D=ROOT/'data/v3_0'


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--index',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID',0)))
    p.add_argument('--draws',type=int,default=8192);p.add_argument('--batch',action='store_true');a=p.parse_args()
    path=Path(a.manifest);cases=json.loads(path.read_text())['cases']
    if a.batch:
        def run(index):
            subprocess.run([sys.executable,str(Path(__file__)),'--manifest',str(path),
                '--index',str(index),'--draws',str(a.draws)],check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run,range(a.index*8,min((a.index+1)*8,len(cases)))))
        return
    case=cases[a.index]
    draw,catalog=int(case['draw']),int(case['catalog'])
    original=D/'shared_recovery'/f'refit_{draw:02}'/f'catalog_{catalog:03}_s2048_native'
    record=json.loads((original/'complete.json').read_text())
    for name,digest in record['outputs'].items():assert sha(original/name)==digest
    for name,digest in record['inputs'].items():
        if name.startswith(('src/','scripts/','data/')):assert sha(ROOT/name)==digest
    out=original.with_name(f'catalog_{catalog:03}_s{a.draws}_native');out.mkdir(exist_ok=True)
    freeze=dict(original_sha256=sha(original/'complete.json'),manifest_sha256=sha(path),
        wrapper_sha256=sha(Path(__file__)),target_acceptance=.99,warmup=1024,draws=a.draws,
        reason=case['reason'],physical_changes=False)
    marker=out/'numerical_refinement.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze
    else:save_json(marker,freeze)
    original_hmc=recovery.hmc
    def refined_hmc(*args,**kwargs):
        kwargs['target_acceptance']=.99
        return original_hmc(*args,**kwargs)
    recovery.hmc=refined_hmc
    sys.argv=[str(ROOT/'scripts/v3_0_shared_recovery.py'),'--draw',str(draw),'--catalog',str(catalog),
              '--draws',str(a.draws),'--warmup','1024','--production']
    if draw in (6,14,23,28):sys.argv+=['--profiles','1024','--quadrature','512']
    recovery.main()


if __name__=='__main__':main()

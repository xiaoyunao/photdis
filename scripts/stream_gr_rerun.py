#!/usr/bin/env python3
"""Preserve the conservative-mask trial and replay only affected imaging stages."""
import subprocess
from photdis.stage1 import require_compute
from stream_common import *


def run(script,*args):
    print('RUN',script,*args,flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],check=True)


def main():
    require_compute();archive=folder('execution')/'before_pure_gr_mask'
    archive.mkdir(exist_ok=True)
    for name in ('local','full_replay','full_inventory','full_identities','full_cache','full_nodes','full_count','full_area','full_atlas','full_rgb','full_validation'):
        p=OUT/name
        if p.exists():
            assert not (archive/name).exists(),('refuse to overwrite earlier trial',name)
            p.rename(archive/name)
    run('stream_gr_mask.py');run('stream_extract.py','audit')
    run('stream_local.py');run('stream_phase.py');run('stream_full.py','replay')
    decision=json.loads((OUT/'full_replay/decision.json').read_text())
    if decision['both_systems_pass']:
        run('stream_full.py','smoke');run('stream_full.py','scan')
        jobs=[subprocess.Popen([sys.executable,str(ROOT/'scripts/stream_full.py'),side]) for side in ('south','north')]
        run('stream_full.py','identities');run('stream_full.py','count')
        for job in jobs:assert job.wait()==0
        run('stream_full.py','assemble')
        run('stream_rgb.py','--cube-stage',str(OUT/'full_atlas'),'--output-stage','full_rgb')
    run('stream_polish.py');run('stream_verify.py')
    if decision['both_systems_pass']:run('stream_full_finish.py')
    run('stream_delivery.py')
    print('PURE_GR_REPLAY_COMPLETE',flush=True)


if __name__=='__main__':main()

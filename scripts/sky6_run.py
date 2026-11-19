#!/usr/bin/env python3
"""Complete all available-sky display and publish it before sequence science."""
import subprocess,argparse
from photdis.stage1 import require_compute
from sky6_common import *


def run(script,*args):
    print('RUN',script,*args,flush=True);subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],check=True)


def main():
    require_compute();parser=argparse.ArgumentParser();parser.add_argument('--resume-at-identities',action='store_true');args=parser.parse_args()
    if args.resume_at_identities:
        for stage in ('ridge_inputs','inventory','area/south','area/north'):assert (OUT/stage/'complete.json').exists(),stage
        areas=[]
    else:
        run('sky6_roster.py');run('sky6_scan.py','smoke')
        areas=[subprocess.Popen([sys.executable,str(ROOT/'scripts/sky6_area.py'),side]) for side in ('south','north')]
        run('sky6_scan.py','scan')
    run('sky6_count.py','identities');run('sky6_count.py','count')
    for p in areas:assert p.wait()==0
    run('sky6_atlas.py','assemble');run('sky6_atlas.py','render');run('sky6_atlas.py','band');run('sky6_prior.py');run('sky6_publish.py')
    print('TASK_A_DELIVERED',flush=True)


if __name__=='__main__':main()

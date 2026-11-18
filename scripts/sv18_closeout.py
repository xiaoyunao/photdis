#!/usr/bin/env python3
"""Run the corrected filter, integration refinement and class control together."""
from concurrent.futures import ThreadPoolExecutor
from photdis.stage1 import require_compute
from sv18_parallel import run

if __name__=='__main__':
    require_compute()
    with ThreadPoolExecutor(max_workers=3) as workers:
        results=list(workers.map(run,('filter_complete','numerics','oracle')))
    if any(results):
        raise SystemExit(1)
    for stage in ('figures','style','verify'):
        if run(stage):
            raise SystemExit(1)

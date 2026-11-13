#!/usr/bin/env python3
"""Run two preflighted structure-entry states with the unchanged 32-worker CLI."""
import argparse
import concurrent.futures
import fcntl
import json
import os
import resource
import subprocess
import sys
import time
from v2_1_near import ROOT, sha, save_json
from photdis.stage1 import require_compute


def main():
    require_compute()
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    draws = plan['draws']
    assert len(draws) == 2 and len(set(draws)) == 2
    assert all(isinstance(i, int) and 0 <= i < 32 for i in draws)
    assert plan['original_writers_terminal_and_absent'] and plan['source_preflight_complete']
    driver = ROOT / 'scripts/v3_1_structure_entry.py'
    assert str(driver.relative_to(ROOT)) in plan['sources']
    expected_payloads = {'data/v3_1/structure/photometry.npz',
                         'data/v3_0/uncertainty/shared_draws.npz'} | {
                             f'data/v3_0/uncertainty_refit/draw_{i:02}/model.joblib' for i in draws}
    assert set(plan['payloads']) == expected_payloads
    assert set(plan['sources']).isdisjoint(plan['payloads'])
    authenticated = {**plan['sources'], **plan['payloads']}
    # Payload hashes are checked here on allocated compute, not by the login
    # node's bounded metadata preflight.
    for name, digest in authenticated.items():
        assert sha(ROOT / name) == digest, name
    memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    declared = int(os.getenv('SLURM_MEM_PER_NODE', '0'))
    if declared > 0:
        memory = min(memory, declared * 1024**2)
    workers = 32 * len(draws)
    # Only the original <=2^21 entry schedule, never the 2^23 refinement.
    # Original 32-worker batches measured ~26 GiB aggregate; allow 3 GiB
    # per worker and require a fresh measured pilot before wider scheduling.
    assert workers * 3 * 1024**3 <= .7 * memory
    assert workers <= len(os.sched_getaffinity(0))
    for draw in draws:
        assert not (ROOT / f'data/v3_1/structure/entry/draw_{draw:02}/complete.json').exists()
    job = os.environ['SLURM_JOB_ID']
    base = ROOT / 'data/v3_1/structure/entry_resource_execution'
    out = base / ('job_' + job)
    out.mkdir(parents=True, exist_ok=False)
    locks = base / 'locks'
    locks.mkdir(exist_ok=True)
    handles = []
    start = time.monotonic()
    record = dict(plan_sha256=sha(path), resource_source_sha256=sha(ROOT / 'scripts/v3_1_structure_entry_pack.py'),
        job=job, draws=draws, processes_per_state=32, total_workers=workers,
        memory_limit_bytes=memory, planning_bound_per_worker_GiB=3,
        results=[], execution_complete=False,
        scope='Resource grouping of two original entry CLIs only. Original integrands, seeds, all512PSF contexts and failed numerical flags remain unchanged; no refinement or scientific acceptance claim.')
    save_json(out / 'inputs.json', record)

    def one(draw):
        log = ROOT / f'logs/photdis_v3_1_entrypack_{job}_draw_{draw:02}.out'
        with log.open('x') as stream:
            result = subprocess.run([sys.executable, '-u', str(driver), '--draw', str(draw)],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        row = dict(draw=draw, exit_code=result.returncode,
                   log=str(log.relative_to(ROOT)), execution_verified=False)
        if result.returncode == 0:
            marker = ROOT / f'data/v3_1/structure/entry/draw_{draw:02}/complete.json'
            done = json.loads(marker.read_text())
            assert done['n'] == 512 and done['all_PSF'] and done['response_draw'] == draw
            assert done['inputs'][str(driver.relative_to(ROOT))] == plan['sources'][str(driver.relative_to(ROOT))]
            assert set(done['contexts']) == {f'context_{i}/complete.json' for i in range(512)}
            for name, digest in {**done['contexts'], **done['outputs']}.items():
                assert sha(marker.parent / name) == digest, name
            row.update(marker=str(marker.relative_to(ROOT)), marker_sha256=sha(marker),
                       numerically_passed=done['passed'], numerical_failures=done['failures'],
                       execution_verified=True)
        return row

    try:
        for draw in draws:
            handle = (locks / f'draw_{draw:02}.lock').open('a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(draws)) as pool:
            futures = [pool.submit(one, draw) for draw in draws]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                record['results'].append(row)
                save_json(out / 'progress.json', record)
                print('EXACT STRUCTURE ENTRY STATE', row['draw'], row['execution_verified'], flush=True)
    finally:
        for handle in handles:
            handle.close()
    for name, digest in authenticated.items():
        assert sha(ROOT / name) == digest, name
    record.update(execution_complete=len(record['results']) == len(draws) and
                  all(r['execution_verified'] for r in record['results']),
                  elapsed_seconds=time.monotonic() - start,
                  maximum_descendant_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    save_json(out / 'complete.json', record)
    assert record['execution_complete'], 'Preserve child failures and checkpoints before any exact resume'


if __name__ == '__main__':
    main()

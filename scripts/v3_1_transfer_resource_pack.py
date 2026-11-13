#!/usr/bin/env python3
"""Execute preflighted, disjoint 2048-draw transfers with the unchanged driver."""
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
    states = plan['states']
    keys = [(r['case'], r['correction']) for r in states]
    assert 1 <= len(states) <= 12 and len(set(keys)) == len(states)
    groups = plan['array_tasks']
    assert 1 <= len(groups) <= 3 and len(set(groups)) == len(groups) and all(0 <= g < 26 for g in groups)
    cases = ['broad', 'candidate', 'holdout']
    expected_keys = [(cases[index // 34], index % 34) for group in groups
                     for index in range(4 * group, min(4 * group + 4, 102))]
    assert set(keys) == set(expected_keys), 'Preserve every state in each replaced array task'
    assert plan['original_writers_terminal_and_absent'] and plan['source_preflight_complete']
    for name, digest in plan['sources'].items():
        assert sha(ROOT / name) == digest
    driver = ROOT / 'scripts/v3_1_heldout_transfer.py'
    assert str(driver.relative_to(ROOT)) in plan['sources']
    memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    declared = int(os.getenv('SLURM_MEM_PER_NODE', '0'))
    if declared > 0:
        memory = min(memory, declared * 1024**2)
    # Pilot is restricted to the measured 2048-draw workload. Do not infer an
    # 8192-draw memory/runtime allowance from this resource experiment.
    assert len(states) * 24 * 1024**3 <= .7 * memory
    assert len(states) <= len(os.sched_getaffinity(0))
    for r in states:
        assert r['case'] in ['broad', 'candidate', 'holdout'] and 0 <= r['correction'] < 34
        suffix = f"{r['case']}/correction_{r['correction']:02}/complete.json"
        assert r['refinement'] == 'data/v3_1/target_refinement/' + suffix
        assert r['output_marker'] == 'data/v3_1/heldout_transfer/' + suffix
        marker = ROOT / r['refinement']
        assert sha(marker) == r['refinement_sha256']
        done = json.loads(marker.read_text())
        assert done['case'] == r['case'] and done['correction'] == r['correction']
        assert done['passed'] and done['selected_draws'] == 2048
        assert not (ROOT / r['output_marker']).exists(), 'Do not overwrite completed provenance'
    job = os.environ['SLURM_JOB_ID']
    base = ROOT / 'data/v3_1/heldout_transfer/resource_execution'
    out = base / ('job_' + job)
    out.mkdir(parents=True, exist_ok=False)
    locks = base / 'locks'
    locks.mkdir(exist_ok=True)
    handles = []
    start = time.monotonic()
    record = dict(plan_sha256=sha(path), resource_script_sha256=sha(ROOT / 'scripts/v3_1_transfer_resource_pack.py'), job=job,
        workers=len(states), memory_limit_bytes=memory, planning_bound_per_worker_GiB=24,
        results=[], execution_complete=False,
        scope='Resource grouping only, unchanged frozen CLI per state. All original outcomes retained; no scientific selection, weight, numerical gate or state normalization change.')
    save_json(out / 'inputs.json', record)

    def one(r):
        name = f"{r['case']}_{r['correction']:02}"
        log = ROOT / f'logs/photdis_v3_1_transferpack_{job}_{name}.out'
        with log.open('x') as stream:
            result = subprocess.run([sys.executable, '-u', str(driver), '--case', r['case'],
                                     '--correction', str(r['correction'])], cwd=ROOT,
                                    stdout=stream, stderr=subprocess.STDOUT)
        row = dict(case=r['case'], correction=r['correction'], exit_code=result.returncode,
                   log=str(log.relative_to(ROOT)), execution_verified=False)
        if result.returncode == 0:
            marker = ROOT / r['output_marker']
            done = json.loads(marker.read_text())
            assert done['case'] == r['case'] and done['correction'] == r['correction']
            assert done['available'] and done['all_PSF'] and not done['references_read']
            assert done['inputs'][r['refinement']] == r['refinement_sha256']
            assert done['inputs'][str(driver.relative_to(ROOT))] == plan['sources'][str(driver.relative_to(ROOT))]
            for name, digest in done['outputs'].items():
                assert sha(marker.parent / name) == digest
            row.update(marker=str(marker.relative_to(ROOT)), marker_sha256=sha(marker),
                       numerically_passed=done['passed'], execution_verified=True)
        return row

    try:
        for case, correction in keys:
            handle = (locks / f'{case}_{correction:02}.lock').open('a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(states)) as pool:
            futures = [pool.submit(one, r) for r in states]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                record['results'].append(row)
                save_json(out / 'progress.json', record)
                print('EXACT TRANSFER STATE', row['case'], row['correction'], row['execution_verified'], flush=True)
    finally:
        for handle in handles:
            handle.close()
    for name, digest in plan['sources'].items():
        assert sha(ROOT / name) == digest
    record.update(execution_complete=all(r['execution_verified'] for r in record['results']) and
                  len(record['results']) == len(states), elapsed_seconds=time.monotonic() - start,
                  maximum_single_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    save_json(out / 'complete.json', record)
    assert record['execution_complete'], 'Preserve child errors and checkpoints before any exact resume'


if __name__ == '__main__':
    main()

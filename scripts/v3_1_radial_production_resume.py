#!/usr/bin/env python3
"""Resume production with original pilot audit settings and unchanged science inputs."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from v2_1_near import ROOT, sha, save_json
from photdis.stage1 import require_compute


def pilot_acceptance():
    records = {}
    for catalog in (0, 48, 112):
        folder = ROOT/'data/v3_1/radial_likelihood'/f'catalog_{catalog:03}_step0.02'
        marker = folder/'complete.json'; record = json.loads(marker.read_text())
        assert record['direct_selection_audit_performed'] and record['selection_passed']
        assert sha(folder/'curves.npz') == record['outputs']['curves.npz']
        records[str(marker.relative_to(ROOT))] = sha(marker)
        for arm in ('broad', 'local'):
            if catalog == 0:
                path = ROOT/'data/v3_1/radial_recovery'/f'catalog_000_{arm}_step0.02_s2048_w512/fine_grid_audit/complete.json'
                assert json.loads(path.read_text())['passed']
            else:
                path = ROOT/'data/v3_1/radial_refinement'/f'catalog_{catalog:03}_{arm}/complete.json'
                result = json.loads(path.read_text()); assert result['passed']
                attempt = result['attempts'][-1]
                assert sha(ROOT/attempt['fine']) == attempt['fine_sha256']
            records[str(path.relative_to(ROOT))] = sha(path)
    return records


def run_catalog(catalog):
    out = ROOT/'data/v3_1/radial_production'; out.mkdir(exist_ok=True)
    log = out/f'catalog_{catalog:03}.log'
    with log.open('a', buffering=1) as stream:
        for step in ('.02', '.01'):
            args = ['--catalog', str(catalog), '--step', step, '--workers', '4']
            if catalog in (0, 48, 112): args += ['--audit-selection']
            subprocess.run([sys.executable, str(ROOT/'scripts/v3_1_radial_likelihood.py'), *args],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        results = {}
        for arm in ('broad', 'local'):
            subprocess.run([sys.executable, str(ROOT/'scripts/v3_1_radial_refine.py'),
                            '--catalog', str(catalog), '--arm', arm],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
            marker = ROOT/'data/v3_1/radial_refinement'/f'catalog_{catalog:03}_{arm}/complete.json'
            record = json.loads(marker.read_text())
            results[arm] = dict(marker=str(marker.relative_to(ROOT)), sha256=sha(marker),
                                passed=record['passed'], selected_draws=record['selected_draws'])
    print('RADIAL PRODUCTION CATALOG', catalog, results, flush=True)
    return dict(catalog=catalog, arms=results)


def main():
    require_compute(); p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, required=True); p.add_argument('--shards', type=int, default=2)
    p.add_argument('--workers', type=int, default=32); a = p.parse_args()
    assert 0 <= a.shard < a.shards and 1 <= a.workers <= 32
    pilots = pilot_acceptance()
    paths = [ROOT/'scripts'/name for name in ['v3_1_radial_production_resume.py', 'v3_1_radial_likelihood.py',
             'v3_1_radial_recovery.py', 'v3_1_radial_refine.py', 'v3_1_radial_quadrature.py']]
    paths += [ROOT/'docs/v3_1_radial_protocol.md']
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    inputs.update(pilot_acceptance=pilots, shard=a.shard, shards=a.shards)
    out = ROOT/'data/v3_1/radial_production'/f'resume_shard_{a.shard}'; out.mkdir(parents=True, exist_ok=True)
    freeze = out/'inputs.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze, inputs)
    catalogs = list(range(a.shard, 176, a.shards)); results = []; failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending = {pool.submit(run_catalog, c): c for c in catalogs}
        for future in concurrent.futures.as_completed(pending):
            try: results.append(future.result())
            except Exception as error:
                failures.append(dict(catalog=pending[future], error=repr(error)))
                print('RADIAL EXECUTION FAILURE', failures[-1], flush=True)
            save_json(out/'progress.json', dict(results=sorted(results, key=lambda x:x['catalog']), failures=failures))
    save_json(out/'complete.json', dict(inputs=inputs, job=os.getenv('SLURM_JOB_ID'), catalogs=catalogs,
        results=sorted(results, key=lambda x:x['catalog']), failures=failures,
        execution_complete=not failures, numerically_passed=sum(all(a['passed'] for a in r['arms'].values()) for r in results),
        scope='All registered scenarios; bounded numerical extensions only. Failed numerical or scientific outcomes retained.'))
    if failures: raise RuntimeError('Catalog execution failures retained; exact resume required')


if __name__ == '__main__': main()

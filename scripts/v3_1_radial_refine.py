#!/usr/bin/env python3
"""Bounded numerical-only extension, preserving each original radial result."""
import argparse
import json
import subprocess
import sys
from v2_1_near import ROOT, sha, save_json
from photdis.stage1 import require_compute


def main():
    require_compute()
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', type=int, required=True)
    p.add_argument('--arm', choices=['broad', 'local'], required=True)
    a = p.parse_args()
    records = []
    for draws in (512, 2048, 8192):
        folder = ROOT/'data/v3_1/radial_recovery'/f'catalog_{a.catalog:03}_{a.arm}_step0.02_s{draws}_w512'
        args = ['--catalog', str(a.catalog), '--arm', a.arm, '--draws', str(draws)]
        subprocess.run([sys.executable, str(ROOT/'scripts/v3_1_radial_recovery.py'), *args], check=True)
        marker = folder/'complete.json'
        record = json.loads(marker.read_text())
        attempt = dict(draws=draws, recovery=str(marker.relative_to(ROOT)), sha256=sha(marker), coarse_passed=record['passed'])
        records.append(attempt)
        if record['passed']:
            subprocess.run([sys.executable, str(ROOT/'scripts/v3_1_radial_quadrature.py'), *args], check=True)
            fine = folder/'fine_grid_audit/complete.json'
            checked = json.loads(fine.read_text())
            attempt.update(fine=str(fine.relative_to(ROOT)), fine_sha256=sha(fine), fine_passed=checked['passed'])
            if checked['passed']:
                break
        print('NUMERICAL EXTENSION', a.catalog, a.arm, draws, attempt, flush=True)
    out = ROOT/'data/v3_1/radial_refinement'/f'catalog_{a.catalog:03}_{a.arm}'
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/'complete.json', dict(catalog=a.catalog, arm=a.arm, attempts=records,
        passed=records[-1].get('fine_passed', False), selected_draws=records[-1]['draws'],
        protocol='Same priors, inputs, seeds and warmup. Fixed512/2048/8192 numerical schedule; no scientific score enters selection. Unresolved flags remain at bound.',
        script_sha256=sha(ROOT/'scripts/v3_1_radial_refine.py')))


if __name__ == '__main__':
    main()

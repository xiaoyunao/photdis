#!/usr/bin/env python3
"""Reuse verified response/MAP for longer chains of the identical target model."""
import argparse
import json
import os
import subprocess
import sys
from v2_1_near import ROOT, sha, save_json
from photdis.stage1 import require_compute

D = ROOT/'data/v3_0'


def main():
    require_compute()
    p = argparse.ArgumentParser()
    p.add_argument('--draw', type=int, required=True)
    p.add_argument('--draws', type=int, default=4096)
    p.add_argument('--profiles', type=int, default=512)
    p.add_argument('--quadrature', type=int, default=256)
    a = p.parse_args()
    source = D/'shared_targets'/f'refit_{a.draw:02}_s512'
    dest = source.with_name(f'refit_{a.draw:02}_s{a.draws}')
    assert dest != source
    record = json.loads((source/'complete.json').read_text())
    assert record['sampler_passed']
    inputs = json.loads((source/'inputs.json').read_text())
    assert inputs['profiles'] == a.profiles and inputs['selection_quadrature'] == a.quadrature
    # Every original physical/source/code input is unchanged, independently of
    # the new chain length. Hard links only share immutable cached inputs.
    for name, value in inputs.items():
        if name.startswith(('data/', 'src/', 'scripts/')):
            assert sha(ROOT/name) == value, name
    dest.mkdir(parents=True, exist_ok=True)
    reused = {}
    for name in ('component_response.npz', 'map.joblib'):
        assert sha(source/name) == record['outputs'][name]
        if (dest/name).exists():
            assert sha(dest/name) == record['outputs'][name]
        else:
            os.link(source/name, dest/name)
        reused[name] = dict(source=str((source/name).relative_to(ROOT)), sha256=record['outputs'][name])
    save_json(dest/'reuse.json', dict(source_complete_sha256=sha(source/'complete.json'), reused=reused,
        purpose='Numerical chain refinement only; identical likelihood, priors, source IDs and MAP metric.'))
    subprocess.run([sys.executable, '-u', str(ROOT/'scripts/v3_0_shared_target.py'), '--draw', str(a.draw),
        '--draws', str(a.draws), '--profiles', str(a.profiles), '--selection-quadrature', str(a.quadrature), '--workers', '64'], check=True)


if __name__ == '__main__':
    main()

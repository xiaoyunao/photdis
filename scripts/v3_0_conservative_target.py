#!/usr/bin/env python3
"""Numerical-only target refinement with conservative warmup and acceptance."""
import argparse
import json
import os
from pathlib import Path
import sys
from v2_1_near import ROOT, sha, save_json
from photdis.stage1 import require_compute
import v3_0_shared_target as target
D = ROOT/'data/v3_0'


def main():
    require_compute()
    parser = argparse.ArgumentParser()
    parser.add_argument('--draw', type=int, required=True)
    parser.add_argument('--draws', type=int, default=1024)
    args = parser.parse_args()
    source = D/'shared_targets'/f'refit_{args.draw:02}_s512'
    destination = source.with_name(f'refit_{args.draw:02}_s{args.draws}')
    assert source != destination
    record = json.loads((source/'complete.json').read_text())
    assert record['sampler_passed']
    original_inputs = json.loads((source/'inputs.json').read_text())
    for name, value in original_inputs.items():
        if name.startswith(('data/', 'src/', 'scripts/')):
            assert sha(ROOT/name) == value, name
    destination.mkdir(exist_ok=True)
    refinement = dict(source_sha256=sha(source/'complete.json'), wrapper_sha256=sha(Path(__file__)),
                      target_acceptance=.99, warmup=1024, draws=args.draws,
                      physical_changes=False, reason='Evidence split tolerance failed; longer nondivergent posterior integration required.')
    marker = destination/'numerical_refinement.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == refinement
    else:
        assert not (destination/'inputs.json').exists()
        save_json(marker, refinement)
    for name in ['component_response.npz', 'map.joblib']:
        assert sha(source/name) == record['outputs'][name]
        if (destination/name).exists():
            assert sha(destination/name) == record['outputs'][name]
        else:
            os.link(source/name, destination/name)
    original_hmc, original_save = target.hmc, target.save_json
    def hmc(*a, **kw):
        kw['target_acceptance'] = .99
        return original_hmc(*a, **kw)
    def save(path, payload):
        if path.name == 'complete.json':
            payload['numerical_refinement'] = refinement
        return original_save(path, payload)
    target.hmc, target.save_json = hmc, save
    sys.argv = [str(ROOT/'scripts/v3_0_shared_target.py'), '--draw', str(args.draw),
                '--draws', str(args.draws), '--warmup', '1024', '--profiles', str(original_inputs['profiles']),
                '--selection-quadrature', str(original_inputs['selection_quadrature']), '--workers', '64']
    target.main()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Numerical-only target refinement preserving exploration length at smaller steps."""
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
    parser.add_argument('--draws', type=int, default=4096)
    args = parser.parse_args()
    assert args.draw == 26 and args.draws == 4096, 'Dedicated diagnosed refinement namespace'
    diagnosis = ROOT/'manifests/v3_0/sampler26_diagnosis.json'
    report = json.loads(diagnosis.read_text())
    assert len(report['rows']) == 3
    assert next(r for r in report['rows'] if r['draws'] == 1024)['minimum_ess'] < 100
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
                      target_acceptance=.95, leapfrog_steps=16, warmup=1024, draws=args.draws,
                      diagnosis_sha256=sha(diagnosis),
                      physical_changes=False, reason='High-acceptance short trajectories mix poorly; extend trajectory and posterior sampling with the physical target unchanged.')
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
        kw['target_acceptance'] = .95
        kw['leapfrog_steps'] = 16
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
    assert json.loads((destination/'complete.json').read_text())['sampler_passed'], 'Sampler failed frozen gate'


if __name__ == '__main__':
    main()

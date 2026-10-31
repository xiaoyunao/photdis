#!/usr/bin/env python3
"""Metadata-only correction; never read or resample the posterior chains."""
import hashlib
import json
from pathlib import Path


def main():
    path = Path('manifests/cluster_rgb_local_color_v1.json')
    original = path.read_bytes()
    result = json.loads(original)
    if 'uniform_control_correction' in result:
        raise RuntimeError('Correction already recorded')
    changed = 0
    for cluster in result['clusters'].values():
        for control in cluster['no_color_identity'].values():
            assert control['maximum_mass_error'] < 1e-12
            assert abs(control['width68'] - .68) < 1e-12
            assert abs(control['width95'] - .95) < 1e-12
            assert abs(control['kl_nats']) < 1e-12
            assert max(control['edge_mass_left'], control['edge_mass_right']) < .15
            control['mode'] = None
            control['edge_flag'] = False
            changed += 1
    assert changed == 8
    result['uniform_control_correction'] = {
        'original_manifest_sha256': hashlib.sha256(original).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'changed_controls': changed,
        'method': 'Only uniform-control mode/edge metadata changed; all posterior values and original run provenance retained.'}
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()

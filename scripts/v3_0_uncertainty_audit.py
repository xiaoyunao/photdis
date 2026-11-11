#!/usr/bin/env python3
"""Same-star one-update/full-update shared-reference sensitivity, not acceptance."""
import json
import os
from pathlib import Path
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, csv_rows
from photdis.stage1 import require_compute
D = ROOT/'data/v3_0'


def read(folder):
    marker = json.loads((folder/'complete.json').read_text()); records = []
    for name, digest in marker['chunks'].items():
        path = ROOT/name if not name.startswith('/') else Path(name)
        assert sha(path) == digest
        meta = json.loads(path.read_text()); data = path.with_suffix('.npz')
        assert sha(data) == meta['summary_sha256']; records.append(load(data))
    return {k: np.concatenate([r[k] for r in records]) for k in records[0]}, sha(folder/'complete.json')


def main():
    require_compute(); records = []; inputs = {}
    for draw in (0, 16):
        one, h1 = read(D/'shared_predictions/application_influence_128'/f'draw_{draw:02}')
        full, h2 = read(D/'shared_predictions/application_refit_128'/f'draw_{draw:02}')
        np.testing.assert_array_equal(one['identity'], full['identity'])
        assert (one['TYPE'] == 'PSF').all() and (full['TYPE'] == 'PSF').all()
        inputs[f'influence_{draw}'] = h1; inputs[f'refit_{draw}'] = h2
        mu = abs(full['mu'][:, 2]-one['mu'][:, 2]); q = abs(full['qnear']-one['qnear'])
        l1 = abs(full['class_distance']-one['class_distance']).sum(axis=(1, 2))
        for group, use in [('all', np.ones(len(mu), bool)), ('supported', full['supported'])]:
            records.append(dict(draw=draw, group=group, n=int(use.sum()), median_abs_delta_mu=float(np.median(mu[use])),
                q95_abs_delta_mu=float(np.quantile(mu[use], .95)), max_delta_qnear=float(q[use].max()),
                q95_delta_qnear=float(np.quantile(q[use], .95)), mean_joint_L1=float(l1[use].mean()),
                q95_joint_L1=float(np.quantile(l1[use], .95))))
    out = D/'uncertainty_refit'; csv_rows(out/'influence_sensitivity.csv', records)
    save_json(out/'influence_sensitivity.json', dict(inputs=inputs, records=records, job=os.getenv('SLURM_JOB_ID'),
        interpretation='One-update approximation versus converged same-family bootstrap at two fixed shared draws; same star IDs and same zero-point/dust perturbations. Numerical differences are not evidence of scientific truth.',
        csv_sha256=sha(out/'influence_sensitivity.csv')))
    print(json.dumps(records), flush=True)


if __name__ == '__main__': main()

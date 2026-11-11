#!/usr/bin/env python3
"""Verify actual reference-prior candidate outputs and record scaling evidence."""
import argparse
import json
import os
import time
import numpy as np
from v2_1_near import ROOT, load, sha, save_json
from photdis.stage1 import require_compute
from v3_0_validation_inputs import check_freeze

D = ROOT/'data/v3_0'


def main():
    require_compute()
    p = argparse.ArgumentParser()
    p.add_argument('--field', type=int, default=int(os.getenv('SLURM_ARRAY_TASK_ID', 0)))
    a = p.parse_args()
    start = time.monotonic()
    method = check_freeze()
    folder = D/'catalog'/f'field_{a.field:02}'
    source = load(folder/'inputs.npz')
    pred = folder/'predictions'
    record = json.loads((pred/'complete.json').read_text())
    assert record['method_freeze'] == method
    rows, converged, chunks, checks = [], 0, 0, 0
    max_joint_error = 0.
    for path in sorted(pred.glob('chunk_*.npz')):
        if path.stem.endswith('_joint'):
            continue
        marker = path.with_suffix('.json')
        assert sha(marker) == record['chunks'][str(marker)]
        meta = json.loads(marker.read_text())
        assert sha(path) == meta['summary_sha256']
        joint = path.with_name(path.stem+'_joint.npz')
        assert sha(joint) == meta['joint_sha256']
        s = load(path)
        np.testing.assert_array_equal(s['identity'], source['identity'][s['rows']])
        np.testing.assert_array_equal(s['band_mask'], source['band_mask'][s['rows']])
        assert (s['TYPE'] == 'PSF').all()
        assert np.isfinite(s['mu']).all() and (np.diff(s['mu'], axis=1) >= 0).all()
        error = float(np.max(np.abs(s['class_distance'].sum(axis=(1, 2))-1)))
        max_joint_error = max(error, max_joint_error)
        assert error < 1e-6
        np.testing.assert_allclose(s['qnear'], s['class_distance'][:, :, :3].sum(axis=(1, 2)), atol=1e-8)
        assert np.isfinite(s['discrepancy_probability']).all()
        rows.extend(s['rows']); converged += int(s['converged'].sum()); chunks += 1; checks += 9
    np.testing.assert_array_equal(rows, np.arange(len(source['identity'])))
    assert len(np.unique(source['identity'])) == len(rows)
    assert (source['TYPE'] == 'PSF').all()
    output_bytes = sum(p.stat().st_size for p in pred.rglob('*') if p.is_file())
    n = len(rows)
    save_json(folder/'reference_audit.json', dict(method_freeze=method, field=a.field, n=n, chunks=chunks,
        checks=checks+3, structural_passed=True, numerically_converged=converged, numerical_failures=n-converged,
        maximum_joint_normalization_error=max_joint_error,
        prediction_elapsed_seconds=record['elapsed_seconds'], actual_output_bytes=output_bytes,
        bytes_per_object=output_bytes/n if n else None,
        production_wall_seconds_per_object=record['elapsed_seconds']/n if n else None,
        prediction_complete_sha256=sha(pred/'complete.json'), source_sha256=sha(folder/'inputs.npz'),
        scope='Actual frozen reference-prior candidate catalog; target transfer, independent scientific calibration and adoption are separate.',
        elapsed_seconds=time.monotonic()-start, job=os.getenv('SLURM_JOB_ID')))
    print('CATALOG REFERENCE AUDIT', a.field, n, converged, output_bytes, flush=True)


if __name__ == '__main__':
    main()

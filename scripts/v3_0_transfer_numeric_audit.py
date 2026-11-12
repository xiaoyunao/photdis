#!/usr/bin/env python3
"""Expose prospective numerical flags without changing predictions or selections."""
import json
import os
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from v3_0_validation_inputs import check_freeze

D = ROOT/'data/v3_0'


def main():
    require_compute()
    method = check_freeze()
    plan_path = D/'catalog/transfer_plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['method_freeze'] == method
    inputs = {str(plan_path.relative_to(ROOT)): sha(plan_path)}
    rows = []
    out = D/'transfer_evaluation/numerical_audit'
    out.mkdir(exist_ok=True)
    for name, expected_n in [('reservation', 3250), ('new_sky', 2048), ('independent_scale', 8192)]:
        base = D/'validation'/name
        folders = [base/'population_transfer'/f'refit_{r["draw"]:02}_s{r["draws"]}'
                   for r in plan['responses']]
        for folder in folders:
            marker = folder/'complete.json'
            record = json.loads(marker.read_text())
            assert record['integration_passed'] and record['rows'] == expected_n
            assert record['inputs']['method_freeze'] == method
            inputs[str(marker.relative_to(ROOT))] = sha(marker)
        ids, errors, reference_ok = [], [], []
        paths = sorted(p.name for p in (folders[0]/'chunks').glob('chunk_*.npz')
                       if not p.stem.endswith('_counts'))
        for filename in paths:
            pieces = []
            for folder in folders:
                path = folder/'chunks'/filename
                record = json.loads(path.with_suffix('.json').read_text())
                assert sha(path) == record['outputs'][filename]
                pieces.append(load(path))
            for piece in pieces:
                np.testing.assert_array_equal(piece['identity'], pieces[0]['identity'])
                assert (piece['TYPE'] == 'PSF').all()
            a = sum(r['retained_weight']*p['half_a'] for r, p in zip(plan['responses'], pieces))
            b = sum(r['retained_weight']*p['half_b'] for r, p in zip(plan['responses'], pieces))
            error = np.max(abs(a-b), axis=(1, 2))+2*plan['discarded_response_mass']
            assert np.isfinite(error).all()
            ids.append(pieces[0]['identity'])
            errors.append(error)
            reference_ok.append(np.logical_and.reduce([p['reference_numerical_converged'] for p in pieces]))
        identity, error, reference = np.concatenate(ids), np.concatenate(errors), np.concatenate(reference_ok)
        source = load(base/'photometry.npz')
        np.testing.assert_array_equal(np.sort(identity), np.sort(source['identity']))
        assert identity.dtype == np.int64 and len(np.unique(identity)) == expected_n
        score_base = D/'transfer_evaluation'/name
        score_marker = score_base/'complete.json'
        scores = json.loads(score_marker.read_text())
        inputs[str(score_marker.relative_to(ROOT))] = sha(score_marker)
        for view in ['matched_reference', 'target']:
            path = score_base/(view+'.npz')
            assert sha(path) == scores['outputs'][path.name]
            np.testing.assert_array_equal(np.sort(load(path)['identity']), np.sort(identity))
        target_ok = error < .005
        atomic_npz(out/(name+'.npz'), identity=identity, final_half_delta=error,
                   reference_numerical_converged=reference, target_mc_converged=target_ok,
                   numerically_converged=reference & target_ok)
        rows.append(dict(cohort=name, n=expected_n, reference_failures=int((~reference).sum()),
                         target_mc_failures=int((~target_ok).sum()),
                         combined_failures=int((~(reference & target_ok)).sum()),
                         maximum_final_half_delta=float(error.max()), tolerance=.005))
    csv_rows(out/'summary.csv', rows)
    save_json(out/'complete.json', dict(method_freeze=method, inputs=inputs,
        structurally_passed=True, rows=sum(r['n'] for r in rows), cohorts=rows,
        numerically_all_passed=all(r['combined_failures'] == 0 for r in rows),
        interpretation='Every prospective object remains in paired scientific scores. These exact-identity flags expose reference quadrature and final response-marginal Monte Carlo limits; no outcome-based exclusion, refit, or threshold relaxation.',
        outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.npz', '.csv')},
        source_sha256=sha(ROOT/'scripts/v3_0_transfer_numeric_audit.py'), job=os.getenv('SLURM_JOB_ID')))
    print('TRANSFER NUMERICAL AUDIT', rows, flush=True)


if __name__ == '__main__':
    main()

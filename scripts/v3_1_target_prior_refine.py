#!/usr/bin/env python3
"""Numerical-only native-grid refinement of an explicitly flagged diagnostic."""
import argparse
import json
import os
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from v3_0_response import observation
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.inference import kernel_loglike
from photdis.v3.systematics import calibration_vectors
from photdis.v3_1.shared_prior import marginalize_native


def main():
    require_compute()
    p = argparse.ArgumentParser(); p.add_argument('--identity', type=int, required=True); a = p.parse_args()
    source = ROOT/'data/v3_1/target_prior/diagnostic'; original = source/f'object_{a.identity}.json'
    done = json.loads((source/'complete.json').read_text())
    assert sha(original) == done['objects'][original.name]
    saved = json.loads(original.read_text()); old = saved['results'][0]
    assert not old['convergence']['converged']
    assert sha(original.with_suffix('.npz')) == saved['pdf_sha256']
    assert sha(source/'shared_parameters.npz') == done['shared_parameters_sha256']
    shared = load(source/'shared_parameters.npz'); previous = load(original.with_suffix('.npz'))
    data = ROOT/'data/v3_0'; table_path = data/'cohorts/development.npz'
    model_path = data/'uncertainty_refit/draw_12/model.joblib'; calibration_path = data/'uncertainty/shared_draws.npz'
    for path in [table_path, model_path, calibration_path]:
        assert sha(path) == done['inputs'][str(path.relative_to(ROOT))]
    t = load(table_path); row = old['row']; assert t['identity'][row] == a.identity and t['TYPE'][row] == 'PSF'
    model = inference_model(joblib.load(model_path)); cal, _ = calibration_vectors(t, load(calibration_path), 12)
    field_row = np.flatnonzero(shared['field'] == old['target_field']); assert len(field_row) == 1
    theta = shared['theta'][:, field_row[0]]; grid = previous['fine_grid']
    old_q, old_probability = previous['fine_mu'], previous['fine_class_distance']
    arrays = {}; levels = []
    for level in (1, 2, 3):
        grid = np.unique(np.r_[grid, (grid[1:]+grid[:-1])/2])
        ll, _ = kernel_loglike(model, observation(t, row), grid, calibration_mag=cal[row])
        result = marginalize_native(model, grid, ll, theta, shared['beta'], shared['weights'])
        quantile = result['posterior'].quantiles(); probability = result['class_distance']
        delta_q = float(abs(quantile-old_q).max()); delta_p = float(abs(probability-old_probability).max())
        passed = bool(delta_q < .03 and delta_p < .003 and result['joint_delta'] < 1e-8
            and abs(result['log_normalization']) < 1e-8 and result['marginal_half_delta'] < .005)
        levels.append(dict(level=level, nodes=len(grid), quantile_delta=delta_q, probability_delta=delta_p,
            joint_delta=result['joint_delta'], log_normalization=result['log_normalization'],
            marginal_half_delta=result['marginal_half_delta'], passed=passed))
        arrays.update({f'level{level}_grid': grid, f'level{level}_logpdf': result['full_logpdf'],
            f'level{level}_mu': quantile, f'level{level}_class_distance': probability})
        old_q, old_probability = quantile, probability
        if passed: break
    arrays.update(identity=np.array([a.identity], 'i8'), TYPE=np.array(['PSF']), draw_ids=shared['draw_ids'], weights=shared['weights'])
    out = source/'refinement'/str(a.identity); out.mkdir(parents=True, exist_ok=True)
    atomic_npz(out/'refinement.npz', **arrays)
    paths = [original, original.with_suffix('.npz'), source/'complete.json', source/'shared_parameters.npz',
        table_path, model_path, calibration_path, ROOT/'scripts/v3_1_target_prior_refine.py', ROOT/'src/photdis/v3_1/shared_prior.py']
    save_json(out/'complete.json', dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in paths},
        outputs={'refinement.npz': sha(out/'refinement.npz')}, identity=a.identity, job=os.getenv('SLURM_JOB_ID'),
        levels=levels, passed=levels[-1]['passed'], original_flag_retained=True,
        original_to_refined_quantile_delta=float(abs(quantile-previous['fine_mu']).max()),
        original_to_refined_probability_delta=float(abs(probability-previous['fine_class_distance']).max()),
        scope='Identical shared parameters, weights, model, calibration and observation; bounded grid bisections only. Original diagnostic and paired report remain unchanged.'))
    print('TARGET PRIOR NUMERICAL REFINEMENT', a.identity, levels, flush=True)


if __name__ == '__main__':
    main()

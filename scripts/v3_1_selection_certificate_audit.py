#!/usr/bin/env python3
"""Real frozen support-certificate equivalence before accelerated entry."""
import json
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3_1.candidate_selection import candidate_component_entry
from photdis.v3_1.exact_support import ExactColorSupport
from photdis.v3_1.certified_support import CertifiedColorSupport
from photdis.v3_1.candidate_quadrature import bounded_component_entry


def main():
    require_compute(); start = time.monotonic(); old = ROOT/'data/v3_1/candidate_selection_pilot/context_1'
    done = json.loads((old/'complete.json').read_text())
    original = old/'q128_seed317101_step0.1.npz'; assert sha(original) == done['outputs'][original.name]
    model_path = ROOT/'data/v3_0/uncertainty_refit/draw_12/model.joblib'; qrf_path = ROOT/'data/v3_0/controls/qrf_grz.joblib'
    paths = [old/'complete.json',original,model_path,qrf_path,ROOT/'scripts/v3_1_selection_certificate_audit.py',
        ROOT/'src/photdis/v3_1/exact_support.py',ROOT/'src/photdis/v3_1/candidate_selection.py',
        ROOT/'src/photdis/v3_1/candidate_quadrature.py',ROOT/'docs/v3_1_selection_refinement.md',ROOT/'src/photdis/v3_1/certified_support.py']
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in paths}; out = ROOT/'data/v3_1/selection_certificate_audit'; out.mkdir(exist_ok=True)
    marker = out/'complete.json'
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == inputs
        return
    model = inference_model(joblib.load(model_path)); qrf = joblib.load(qrf_path); source = load(original)
    support = CertifiedColorSupport(qrf); rng=np.random.default_rng(319601)
    colors=rng.uniform(qrf.lo,qrf.hi,size=(1000000,2)); magnitudes=np.c_[20+colors[:,0],np.full(len(colors),20.),20-colors[:,1]]
    np.testing.assert_allclose(qrf.feature_matrix(magnitudes),colors,atol=1e-13,rtol=0)
    np.testing.assert_array_equal(support(magnitudes),ExactColorSupport(qrf)(magnitudes))
    response = candidate_component_entry(model,np.array(done['raw_error']),np.array(done['transmission']),np.array(done['calibration_mag']),
        source['grid'],support=support,draws=128,seed=317101)
    difference = abs(response-source['binary_component_entry']); passed = bool(difference.max() < 1e-14)
    pruned,bound = bounded_component_entry(model,np.array(done['raw_error']),np.array(done['transmission']),np.array(done['calibration_mag']),
        source['grid'],support=support,draws=128,seed=317101)
    pruning_delta = abs(pruned-response)
    passed = passed and bool((pruning_delta <= bound+1e-13).all())
    atomic_npz(out/'comparison.npz',maximum_by_binary_kernel=difference.max(axis=2))
    save_json(marker,dict(inputs=inputs,outputs={'comparison.npz':sha(out/'comparison.npz')},job=os.getenv('SLURM_JOB_ID'),
        certified_queries=support.certified,fallback_queries=support.fallback,random_points_verified=1000000,
        max_probability_difference=float(difference.max()),max_pruning_delta=float(pruning_delta.max()),
        max_pruning_bound=float(bound.max()),passed=passed,original_quadrature_passed=done['passed'],elapsed_seconds=time.monotonic()-start,
        scope='Identical Gaussian QMC nodes and original QRF support; 1-Lipschitz certified interior decisions with exact ambiguous-boundary fallback. No response/model/selection cut change.'))
    assert passed


if __name__ == '__main__': main()

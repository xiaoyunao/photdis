#!/usr/bin/env python3
"""Explicit conditional transfer of the sampled target population to new stars.

This product is q(shared | the 26000-object target sample) times the product of
per-star conditional latent posteriors. It propagates the same shared draws
through all objects, but does NOT refit the population on the remaining catalog.
The exact observed-sample joint posterior and design-based totals are separate.
Do not relabel this conditional transfer as a full-catalog Bayesian refit.
"""
import argparse
import concurrent.futures
import json
import hashlib
import multiprocessing as mp
import os
import resource
from pathlib import Path
import time
import joblib
import numpy as np
from scipy.special import softmax
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.hierarchy import FieldPopulation
from photdis.v3.systematics import calibration_vectors
from photdis.v3.native_response import native_response
from photdis.v3.ensemble_counts import merge_chunk_counts
from photdis.v3.population import SharedCounts
from v3_0_validation_inputs import check_freeze
from v3_0_response import observation
import v3_0_predict as predict
import v3_0_shared_predict as shared
import v3_0_shared_products as products

D = ROOT/'data/v3_0'
_LATENT_SEED = None


class ProjectionCounts(SharedCounts):
    def __init__(self, *args, seed=None, **kwargs):
        # Common parameter IDs induce physical covariance. Independent fields
        # must not acquire extra covariance from reusing latent random numbers.
        super().__init__(*args, seed=np.random.SeedSequence([seed or 0, _LATENT_SEED]), **kwargs)


def product_chunk(path):
    output = products._OUT/'chunks'/path.name
    marker = output.with_suffix('.json')
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get('projection_flags_exported'):
            assert saved['inputs'] == products._INPUTS
            for name, digest in saved['outputs'].items(): assert sha(output.parent/name) == digest
            return str(marker)
    summary = load(path)
    packed = load(path.with_name(path.stem+'_joint.npz'))
    values = []
    for j, row in enumerate(summary['rows']):
        lo, hi = packed['full_grid_offset'][j:j+2]
        grid = np.unique(np.r_[packed['full_mu'][lo:hi], np.linspace(-20., 50., 1401)])
        values.append(native_response(products._MODEL, observation(products._SOURCE, row), grid,
            packed['source_log_evidence'][j], calibration_mag=products._CALIBRATION[row]))
    products._RESPONSE = dict(identity=summary['identity'], rows=summary['rows'], response=np.array(values))
    result = products.chunk(path)
    payload = load(output)
    payload.update(photometric_system=products._SOURCE['photometric_system'][summary['rows']],
        negative_intrinsic_support=summary['negative_intrinsic_support'],
        reference_mu=summary['mu'], reference_qnear=summary['qnear'],
        prior_delta_qnear=payload['qnear']-summary['qnear'],
        prior_delta_mu50=payload['mu'][:,2]-summary['mu'][:,2],
        prior_sensitive=np.abs(payload['qnear']-summary['qnear'])>.05,
        extrapolation=~payload['supported'], multimodal=(payload['mode_mass']>.05).sum(axis=1)>1)
    atomic_npz(output, **payload)
    saved = json.loads(marker.read_text());saved['outputs'][output.name]=sha(output)
    saved['projection_flags_exported']=True
    saved['flag_scope']='Prior sensitivity means absolute qnear change >0.05; multimodal means at least two modes of mass >0.05. Descriptive flags, not externally calibrated quality cuts.'
    save_json(marker,saved)
    return result


def main():
    global _LATENT_SEED
    require_compute()
    p = argparse.ArgumentParser()
    p.add_argument('scope', choices=['catalog', 'validation'])
    p.add_argument('name')
    p.add_argument('--draw', type=int, required=True)
    p.add_argument('--draws', type=int, default=512)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--workers', type=int, default=64)
    a = p.parse_args()
    start = time.monotonic()
    method = check_freeze()
    if a.scope == 'catalog':
        folder = D/'catalog'/f'field_{int(a.name):02}'
        source, source_marker = folder/'inputs.npz', folder/'prepared.json'
        assert json.loads(source_marker.read_text())['outputs']['inputs.npz'] == sha(source)
    else:
        assert a.name in ('reservation', 'new_sky', 'independent_scale')
        folder = D/'validation'/a.name
        source, source_marker = folder/'photometry.npz', folder/'photometry.json'
        meta = json.loads(source_marker.read_text())
        assert meta['method_freeze'] == method and meta['sha256'] == sha(source)
    model_path = D/'uncertainty_refit'/f'draw_{a.draw:02}/model.joblib'
    target = D/'shared_targets'/f'refit_{a.draw:02}_s{a.draws}'
    native = D/'native_responses'/f'refit_{a.draw:02}_s{a.draws}'
    nr = json.loads((native/'complete.json').read_text())
    assert nr['passed'] and nr['source_target_sha256'] == sha(target/'complete.json')
    correction = load(native/'chain_correction.npz')
    assert sha(native/'chain_correction.npz') == nr['outputs']['chain_correction.npz']
    data = load(source)
    if a.limit:
        data = {k: v[:a.limit] for k, v in data.items()}
    assert (data['TYPE'] == 'PSF').all()
    shared_path = D/'uncertainty/shared_draws.npz'
    calibration, dust = calibration_vectors(data, load(shared_path), a.draw)
    model = inference_model(joblib.load(model_path))
    output = folder/'population_transfer'/f'refit_{a.draw:02}_s{a.draws}'
    if a.limit:
        output = output.with_name(output.name+f'_pilot{a.limit}')
    output.mkdir(parents=True, exist_ok=True)
    pred = output/'reference_prediction'
    pred.mkdir(exist_ok=True)
    control = D/'controls/qrf_grz.joblib'
    paths = [source, source_marker, model_path, control, shared_path, target/'complete.json',
        native/'complete.json', native/'chain_correction.npz', D/'cohorts/application.npz',
        ROOT/'scripts/v3_0_projection.py', ROOT/'scripts/v3_0_predict.py',
        ROOT/'scripts/v3_0_shared_products.py', ROOT/'scripts/v3_0_shared_predict.py']
    inputs = {str(x.relative_to(ROOT)): sha(x) for x in paths}
    inputs.update(vars(a), method_freeze=method)
    if (output/'inputs.json').exists():
        assert json.loads((output/'inputs.json').read_text()) == inputs
    else:
        save_json(output/'inputs.json', inputs)
    # Native reference prediction uses the same response and calibration draw.
    predict._TABLE, predict._MODEL, predict._CONTROL = data, model, joblib.load(control)
    predict._FROZEN, predict._DEST, predict._BANDS = inputs, pred, np.ones(6, bool)
    shared._CALIBRATION = calibration
    predict.observation, predict.infer_reference = shared.shared_observation, shared.shared_infer
    predict.reference_log_density = shared.shared_log_density
    tasks = [np.arange(i, min(i+32, len(data['identity']))) for i in range(0, len(data['identity']), 32)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('fork')) as pool:
        markers = list(pool.map(predict.chunk, tasks))
    reference_converged=sum(json.loads(Path(p).read_text())['converged'] for p in markers)
    save_json(pred/'complete.json', dict(inputs=inputs, rows=len(data['identity']), reference_converged=reference_converged, chunks={p: sha(p) for p in markers}))
    # Extend field random effects from the frozen hypermean for genuinely new sky.
    original_fields = np.unique(load(D/'cohorts/application.npz')['field'])
    dummy = np.ones((len(original_fields), len(model.weights), 5))
    hierarchy = FieldPopulation(dummy, dummy, model, original_fields)
    z = correction['samples']
    decoded = [hierarchy.decode(row) for row in z]
    fields = np.unique(data['field'])
    theta = np.empty((len(z), len(fields), 3, 5))
    beta = np.array([x[1] for x in decoded])
    global_logits = np.array([x[3] for x in decoded])
    for j, field in enumerate(fields):
        if field in original_fields:
            index = int(np.flatnonzero(original_fields == field)[0])
            theta[:, j] = np.array([x[0][index] for x in decoded])
        else:
            rng = np.random.default_rng(np.random.SeedSequence([316001, a.draw, int(field)+1000000]))
            local = global_logits+hierarchy.field_sigma*rng.normal(size=global_logits.shape)
            theta[:, j] = softmax(np.c_[local, np.zeros(len(local))], axis=1).reshape(-1, 3, 5)
    products._MODEL, products._SOURCE, products._FIELDS = model, data, fields
    products._THETA, products._BETA, products._WEIGHTS = theta, beta, correction['weights']
    products._DRAW_IDS, products._CALIBRATION, products._NATIVE = correction['draw_ids'], calibration, True
    products._OUT, products._PRED, products._INPUTS = output, pred, inputs
    _LATENT_SEED = int(hashlib.sha256(f'{a.scope}/{a.name}/{a.draw}'.encode()).hexdigest()[:16],16)
    products.SharedCounts = ProjectionCounts
    (output/'chunks').mkdir(exist_ok=True)
    paths = sorted(x for x in pred.glob('chunk_*.npz') if not x.stem.endswith('_joint'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('fork')) as pool:
        markers = list(pool.map(product_chunk, paths))
    records = [load(output/'chunks'/(Path(p).stem+'_counts.npz')) for p in markers]
    counts = merge_chunk_counts(records)
    atomic_npz(output/'counts.npz', **counts)
    atomic_npz(output/'shared_parameters.npz', draw_ids=correction['draw_ids'], weights=correction['weights'],
        theta=theta, beta=beta, fields=fields, calibration_mag=calibration, dust_offset=dust)
    metadata = [json.loads(Path(p).read_text()) for p in markers]
    errors = {k: max(x[k] for x in metadata) for k in ('maximum_native_integral_error', 'maximum_integral_error', 'maximum_joint_delta', 'maximum_omitted_bound')}
    save_json(output/'complete.json', dict(inputs=inputs, rows=len(data['identity']), global_response_draw=a.draw,
        integration_passed=all(v < .001 for v in errors.values()), reference_converged=reference_converged, errors=errors,
        conditional_marginal_half_delta=counts['marginal_half_delta'],
        counts_sha256=sha(output/'counts.npz'), shared_parameters_sha256=sha(output/'shared_parameters.npz'),
        target_complete_sha256=sha(target/'complete.json'), chunks={p: sha(p) for p in markers},
        scope='Conditional transferred joint distribution q(shared|26000 sampled targets) times per-star conditional posteriors. Shared uncertainty retained, but population is not refitted on these new objects. Not a full-catalog Bayesian update or intrinsic density estimate.',
        numerical_acceptance='Final response-marginal precision must be checked after combining response draws.',
        elapsed_seconds=time.monotonic()-start,
        output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        maximum_process_rss_kib=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        rss_scope='Maximum individual process RSS, not sum of concurrent worker RSS; Slurm accounting is a separate job-level check.',
        job=os.getenv('SLURM_JOB_ID')))
    print('CONDITIONAL TRANSFER', a.scope, a.name, a.draw, len(data['identity']), errors, flush=True)


if __name__ == '__main__':
    main()

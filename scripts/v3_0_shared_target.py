#!/usr/bin/env python3
"""Conditional target fits and bridge evidence for the finite shared response prior."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.compression import unpack
from photdis.v3.target import component_response
from photdis.v3.fast_hierarchy import FastFieldPopulation
from photdis.v3.sampling import hmc, chain_diagnostics
from photdis.v3.selection import effective_errors
from photdis.v3.selection_table import SelectionTable
from photdis.v3.systematics import calibration_vectors
from photdis.v3.evidence import bridge_integral
D = ROOT/'data/v3_0'
_MODEL = _OUT = _HIERARCHY = _FIT = _ARGS = None


def response_chunk(path):
    path = Path(path); summary_path = path.with_name(path.name.replace('_joint', ''))
    marker = json.loads(summary_path.with_suffix('.json').read_text())
    output = _OUT/'response_chunks'/summary_path.name; output.parent.mkdir(exist_ok=True)
    if output.exists():
        old = json.loads(output.with_suffix('.json').read_text())
        assert old['input_joint'] == sha(path) and old['sha256'] == sha(output)
        return str(output)
    assert sha(path) == marker['joint_sha256'] and sha(summary_path) == marker['summary_sha256']
    packed = load(path); summary = load(summary_path)
    response = []; evidence = []
    for i in range(len(summary['identity'])):
        record = unpack(packed, i); response.append(component_response(record, _MODEL).astype('f4'))
        evidence.append(record['source_log_evidence'])
    atomic_npz(output, response=np.array(response), identity=summary['identity'], reference_log_evidence=evidence)
    save_json(output.with_suffix('.json'), dict(input_joint=sha(path), sha256=sha(output)))
    return str(output)


def sample_chain(i):
    path = _OUT/f'chain_{i}.joblib'
    if path.exists(): return joblib.load(path)
    record = hmc(_HIERARCHY.objective, _FIT['map'], _FIT['metric'], draws=_ARGS.draws,
        warmup=_ARGS.warmup, seed=306000+_ARGS.draw*100+i)
    joblib.dump(record, path.with_suffix('.tmp')); path.with_suffix('.tmp').replace(path)
    return record


def main():
    global _MODEL, _OUT, _HIERARCHY, _FIT, _ARGS
    require_compute(); p = argparse.ArgumentParser()
    p.add_argument('--draw', type=int, default=int(os.getenv('SLURM_ARRAY_TASK_ID', 0)))
    p.add_argument('--ensemble', choices=['influence', 'refit'], default='refit')
    p.add_argument('--draws', type=int, default=512); p.add_argument('--warmup', type=int, default=512)
    p.add_argument('--workers', type=int, default=32); p.add_argument('--profiles', type=int, default=256)
    args = _ARGS = p.parse_args(); start = time.monotonic()
    prediction = D/'shared_predictions'/f'application_{args.ensemble}_0'/f'draw_{args.draw:02}'
    pred_complete = prediction/'complete.json'; assert pred_complete.exists()
    model_path = D/'uncertainty/ensemble.joblib' if args.ensemble == 'influence' else D/'uncertainty_refit'/f'draw_{args.draw:02}/model.joblib'
    fitted = joblib.load(model_path)
    if args.ensemble == 'influence': fitted = fitted[args.draw]
    else:
        record = json.loads(model_path.with_name('complete.json').read_text())
        assert record['converged'] and sha(model_path) == record['sha256']
    _MODEL = inference_model(fitted)
    selection_path = D/'shared_selection'/f'{args.ensemble}_{args.draw:02}_p{args.profiles}/table.npz'
    selection_record = json.loads(selection_path.with_name('complete.json').read_text())
    assert selection_record['passed'] and selection_record['sha256'] == sha(selection_path)
    table = SelectionTable(**load(selection_path))
    source_path = D/'cohorts/application.npz'; source = load(source_path)
    shared_path = D/'uncertainty/shared_draws.npz'; shared = load(shared_path)
    out = _OUT = D/'shared_targets'/f'{args.ensemble}_{args.draw:02}_s{args.draws}'; out.mkdir(parents=True, exist_ok=True)
    paths = [model_path, source_path, shared_path, pred_complete, selection_path, prediction/'shared_system.npz', ROOT/'scripts/v3_0_shared_target.py']
    paths += [ROOT/'src/photdis/v3'/x for x in ('target.py', 'hierarchy.py', 'fast_hierarchy.py', 'compression.py', 'sampling.py', 'evidence.py', 'systematics.py')]
    inputs = {str(x.relative_to(ROOT)): sha(x) for x in paths}; inputs.update(vars(args))
    if (out/'inputs.json').exists(): assert json.loads((out/'inputs.json').read_text()) == inputs
    else: save_json(out/'inputs.json', inputs)
    if (out/'complete.json').exists():
        old = json.loads((out/'complete.json').read_text()); assert old['inputs'] == inputs
        for name, digest in old['outputs'].items(): assert sha(out/name) == digest
        return
    response_path = out/'component_response.npz'
    if not response_path.exists():
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
            names = list(pool.map(response_chunk, sorted(prediction.glob('chunk_*_joint.npz'))))
        pieces = [load(name) for name in names]; ids = np.concatenate([x['identity'] for x in pieces])
        order = np.argsort(source['identity']); rows = order[np.searchsorted(source['identity'][order], ids)]
        np.testing.assert_array_equal(source['identity'][rows], ids)
        calibration, _ = calibration_vectors({k: v[rows] for k, v in source.items()}, shared, args.draw)
        recorded = load(prediction/'shared_system.npz')
        np.testing.assert_array_equal(recorded['identity'], ids)
        np.testing.assert_allclose(calibration, recorded['calibration_mag'], rtol=0, atol=0)
        errors = effective_errors({k: v[rows] for k, v in source.items()})*10**(.4*calibration[:, [0, 1, 3]])
        alpha, audit = table.evaluate(errors)
        if not audit['inside_profile_hull'].all(): raise RuntimeError('Uncertified shared target selection')
        atomic_npz(response_path, response=np.concatenate([x['response'] for x in pieces]), selection=alpha.astype('f4'),
            reference_log_evidence=np.concatenate([x['reference_log_evidence'] for x in pieces]), identity=ids, rows=rows)
    r = load(response_path); rows = r['rows']; assert (source['TYPE'][rows] == 'PSF').all()
    _HIERARCHY = FastFieldPopulation(r['response'].sum(axis=3), r['selection'], _MODEL,
                                    source['field'][rows], source['design_weight'][rows])
    if (out/'map.joblib').exists(): _FIT = joblib.load(out/'map.joblib')
    else: _FIT = _HIERARCHY.fit(); joblib.dump(_FIT, out/'map.joblib')
    print('SHARED TARGET MAP', args.draw, _FIT['objective'], flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('fork')) as pool:
        records = list(pool.map(sample_chain, range(4)))
    samples = np.array([x['samples'] for x in records]); diagnostic = chain_diagnostics(samples)
    atomic_npz(out/'chains.npz', samples=samples, rhat=diagnostic['rhat'], ess=diagnostic['ess'])
    # Standard-normal coordinates have a proper N(0,I) prior; its normalization
    # and the identical case-weighted reference evidence are BOTH indispensable.
    weights = source['design_weight'][rows]/source['design_weight'][rows].mean()
    constant = float(weights@r['reference_log_evidence'])
    def logdensity(z): return -_HIERARCHY.objective(z)[0]-.5*_HIERARCHY.dimension*np.log(2*np.pi)
    bridge = bridge_integral(logdensity, samples.transpose(1, 0, 2).reshape(-1, samples.shape[-1]),
                             proposal_draws=2048, seed=307000+args.draw)
    atomic_npz(out/'bridge_diagnostics.npz', posterior_logratio=bridge.pop('log_ratio_posterior'),
               proposal_logratio=bridge.pop('log_ratio_proposal'))
    bridge['log_evidence'] = bridge['log_integral']+constant; bridge['weighted_reference_constant'] = constant
    save_json(out/'bridge.json', bridge)
    sampler_passed = diagnostic['max_rhat'] < 1.05 and diagnostic['minimum_ess'] >= 100 and not sum(x['divergent'] for x in records)
    passed = sampler_passed and bridge['fixed_point_converged'] and bridge['proposal_bridge_ess'] >= 100 and bridge['half_log_integral_difference'] < .2
    save_json(out/'complete.json', dict(inputs=inputs, n=len(rows), global_draw_id=args.draw,
        max_rhat=diagnostic['max_rhat'], minimum_ess=diagnostic['minimum_ess'], passed=bool(passed), sampler_passed=bool(sampler_passed),
        bridge=bridge, conditional_scope='Finite same-family bootstrap/calibration/system response prior; no between-family or independent absolute-scale uncertainty',
        weight_scope='Design-rescaled generalized likelihood; actual sample size, not expanded parent count',
        outputs={p.name: sha(p) for p in out.iterdir() if p.suffix in ('.npz', '.joblib')}, elapsed_seconds=time.monotonic()-start, job=os.getenv('SLURM_JOB_ID')))
    print('SHARED TARGET COMPLETE', args.draw, passed, flush=True)


if __name__ == '__main__': main()
